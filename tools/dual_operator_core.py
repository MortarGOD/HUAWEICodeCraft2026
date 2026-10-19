"""V compensation between predicted raw and deployed-quantized attention operators."""
_DO_ROUNDS = 4
_DO_MSE_CAP = 1.5
DO_DEBUG = {}
_do_parent_calibration = hif4_calibration_attention
_do_parent_v = hif4_dynamic_quantize_v


def _do_fit(samples, states, qh, kh, d):
    selected = [s for s in samples[:-1] if 4 <= s['v'][0].shape[0] <= 4096][-2:]
    if len(selected) != 2:
        return None
    arrays = {r: [] for r in ('q', 'k', 'v', 'aq', 'ak')}
    groups = []
    count = 0
    for s in selected:
        n = s['v'][0].shape[0]
        ix = torch.linspace(0, n-1, min(128, n)).long()
        for r in 'qkv':
            arrays[r].append(dequantize_nvfp4(s[r][0][ix], s[r][1][ix]).float())
        arrays['aq'].append(_joint_quant_queries(s, states['q_state'], qh, kh, d, ix))
        ak = _deq_params(hif4_dynamic_quantize_k(*s['k'], kh, d, states['k_state']))
        # A per-sequence key constant is softmax-null. Remove it before fitting.
        arrays['k'][-1] -= arrays['k'][-1].mean(0)
        arrays['ak'].append(ak[ix] - ak[ix].mean(0))
        groups.append((count, count+ix.numel()))
        count += ix.numel()
    arrays = {r: torch.cat(v) for r, v in arrays.items()}
    v = arrays['v']; center = v.mean(0); vc = v-center
    rank = min(64, v.shape[0]-1, v.shape[1])
    if v.shape[1] > v.shape[0]:
        eig, left = torch.linalg.eigh(vc@vc.T/v.shape[0])
        floor = eig[-1].clamp_min(1e-12)*1e-8
        basis = vc.T@left[:, -rank:] / (v.shape[0]*eig[-rank:].clamp_min(floor)).sqrt()[None]
    else:
        eig, basis = torch.linalg.eigh(vc.T @ vc / v.shape[0])
    projection = basis[:, -rank:] / (eig[-rank:].clamp_min(0)+eig[-1].clamp_min(1e-12)*.01).sqrt()[None]
    st = dict(center=center.to(torch.bfloat16).contiguous(), projection=projection.to(torch.bfloat16).contiguous())
    f = _vk_features(v, st)
    chol = torch.linalg.cholesky(f.T@f + torch.eye(f.shape[1])*(.03*f.shape[0]))
    coefficients = {}
    for r, nh in (('q', qh), ('k', kh), ('aq', qh), ('ak', kh)):
        coefficients[r] = torch.cholesky_solve(f.T@arrays[r], chol).reshape(-1, nh, d).permute(1, 0, 2)
    kernels = []
    for qr, kr in (('q', 'k'), ('aq', 'ak')):
        bq = coefficients[qr]; bk = coefficients[kr].repeat_interleave(qh//kh, 0)
        kernels.append((bq@bk.transpose(1, 2))/(d**.5))
    st['raw_kernel'], st['quant_kernel'] = [x.contiguous() for x in kernels]
    st['modes'] = _vu_modes(f, arrays['q'], arrays['k'], groups, qh, kh, d).to(torch.bfloat16).contiguous()
    st['active'] = torch.ones(kh, dtype=torch.bool)
    if not all(bool(torch.isfinite(v).all()) for v in st.values()):
        return None
    return st


def _do_probabilities(x, st, kh):
    f = _vk_features(x, st)
    qh = st['raw_kernel'].shape[0]
    nq = max(1, min(4, 128//max(1, (qh//kh)*4)))
    ix = torch.linspace(0, x.shape[0]-1, min(nq, x.shape[0])).long()
    modes = st['modes'].float()@f.T * (2.**.5)
    offsets = torch.cat((modes, -modes), 1)
    out = []
    for key in ('raw_kernel', 'quant_kernel'):
        logits = (f[ix][None]@st[key].float())@f.T
        p = torch.softmax(logits[:, :, None]+offsets[:, None], -1)
        out.append(p.reshape(kh, -1, x.shape[0]).contiguous())
    return tuple(out)


def _do_continuous(x, raw, quant, ridge):
    target = raw@x
    gram = quant@quant.transpose(1, 2)
    system = gram + ridge[:, None, None]*torch.eye(gram.shape[-1], dtype=gram.dtype)
    correction = torch.linalg.solve(system, target-quant@x)
    return x + quant.transpose(1, 2)@correction


def _do_refine(x, parent, st, kh, d):
    if not 4 <= x.shape[0] <= 4096 or not bool(st['active'].any()):
        return parent
    raw, quant = _do_probabilities(x, st, kh)
    n = x.shape[0]; rows = quant.shape[1]
    xh = x.reshape(n, kh, d).transpose(0, 1).contiguous()
    base = _deq_params(parent).reshape(n, kh, d).transpose(0, 1).contiguous()
    step = (_params_unit_flat(parent)*.25).reshape(n, kh, d).transpose(0, 1).contiguous()
    reference_mse = (xh-base).square().mean(1).clamp_min(1e-30)
    ridge = rows/(n*n) + .05*quant.square().sum(2).mean(1)
    target = raw@xh
    def cost(y):
        mse = (xh-y).square().mean(1)
        return (target-quant@y).square().mean(1) + (ridge*n/rows)[:, None]*mse, mse
    best = base; best_cost, _ = cost(base)
    center = _do_continuous(xh, raw, quant, ridge)
    cap = reference_mse.sqrt()[:, None]*2
    center = xh+(center-xh).clamp(-cap, cap)
    shift = center-xh
    mass = quant.sum(1).clamp_min(1e-12)[:, :, None] + ridge[:, None, None]
    for _ in range(_DO_ROUNDS):
        proposal = torch.round((xh+shift)/step).clamp(-7, 7)*step
        value, mse = cost(proposal)
        take = (value < best_cost) & (mse <= reference_mse*_DO_MSE_CAP) & st['active'][:, None]
        best = torch.where(take[:, None], proposal, best)
        best_cost = torch.where(take, value, best_cost)
        gradient = quant.transpose(1, 2)@(target-quant@proposal) + ridge[:, None, None]*(xh-proposal)
        shift = shift + .75*gradient/mass
    result = _values_to_params(best.transpose(0, 1).reshape(x.shape).contiguous(), parent)
    blocks = st['active'].repeat_interleave(d//64)
    return {k: torch.where(blocks.reshape(1, -1, *([1]*(v.ndim-2))), v, parent[k]) for k, v in result.items()}


def _do_guard(samples, parent, qh, kh, d):
    DO_DEBUG.clear(); DO_DEBUG.update(accepted=0, eligible=False, error=False)
    if len(samples)<3 or d%64 or d>256 or qh>128 or qh//kh>32 or not all(parent[r+'_state'].get('gq') for r in 'qk'):
        return parent
    hold = samples[-1]; n = hold['v'][0].shape[0]
    if not 4 <= n <= 4096:
        return parent
    st = _do_fit(samples, parent, qh, kh, d)
    if st is None:
        return parent
    DO_DEBUG['eligible'] = True
    ix = torch.linspace(0, n-1, min(64, n)).long()
    raw = [dequantize_nvfp4(*hold[r]).float() for r in 'qkv']
    reference = _joint_output(*raw, qh, kh, d, ix)
    q = _joint_quant_queries(hold, parent['q_state'], qh, kh, d, ix)
    k = _deq_params(hif4_dynamic_quantize_k(*hold['k'], kh, d, parent['k_state']))
    base = _do_parent_v(*hold['v'], kh, d, parent['v_state'])
    trial = _do_refine(raw[2], base, st, kh, d)
    costs = []
    for p in (base, trial):
        error = (_joint_output(q, k, _deq_params(p), qh, kh, d, ix, query_subset=True)-reference).square().mean((-1, -2))
        costs.append(torch.stack((error[::2].mean(0), error[1::2].mean(0))))
    mask = (costs[1] < costs[0]*.995).all(0) & torch.isfinite(costs[1]).all(0)
    DO_DEBUG.update(accepted=int(mask.sum()), ratios=(costs[1].mean(0)/costs[0].mean(0).clamp_min(1e-30)).tolist(),
                    changed_values=int((_deq_params(trial)!=_deq_params(base)).sum()))
    if not bool(mask.any()):
        return parent
    st['active'] = mask
    return dict(parent, v_state=dict(dual_operator=st, fallback=parent['v_state']))


@torch.inference_mode()
def hif4_calibration_attention(samples, qh, kh, d):
    parent = _do_parent_calibration(samples, qh, kh, d)
    try:
        return _do_guard(samples, parent, qh, kh, d)
    except (RuntimeError, ValueError) as exc:
        DO_DEBUG.update(error=True, error_type=type(exc).__name__)
        return parent


def hif4_dynamic_quantize_v(q, s, kh, d, state):
    if not isinstance(state, dict) or 'dual_operator' not in state:
        return _do_parent_v(q, s, kh, d, state)
    parent = _do_parent_v(q, s, kh, d, state['fallback'])
    if not 4 <= q.shape[0] <= 4096:
        return parent
    try:
        return _do_refine(dequantize_nvfp4(q, s).float(), parent, state['dual_operator'], kh, d)
    except (RuntimeError, ValueError):
        return parent
