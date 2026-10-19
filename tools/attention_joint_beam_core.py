"""v395: bounded joint metadata / correlated-code Q search."""
def _jb_beam(center, step, u, width):
    # center: [batch, rows, block]. Keep a true beam of partial assignments.
    batch, rows, d = center.shape
    work = center[:, :, None, :].clone()
    codes = torch.zeros_like(work)
    costs = torch.zeros(batch, rows, 1, dtype=center.dtype)
    for i in range(d):
        current = work[..., i]
        lower = torch.floor(current/step[:, :, None, i]).clamp(-7, 7)
        upper = (lower+1).clamp(-7, 7)
        # At saturation both branches may duplicate, but never leave the grid.
        proposed = torch.stack((lower, upper), -1)*step[:, :, None, None, i]
        err = (current[..., None]-proposed)/u[:, None, None, None, i, i]
        trial_cost = (costs[..., None]+err.square()).flatten(-2)
        keep = min(width, trial_cost.shape[-1])
        cost, selected = torch.topk(trial_cost, keep, dim=-1, largest=False, sorted=True)
        previous = selected//2
        selected_work = work.gather(2, previous[..., None].expand(-1, -1, -1, d))
        selected_codes = codes.gather(2, previous[..., None].expand(-1, -1, -1, d))
        err = err.flatten(-2).gather(-1, selected)
        q = proposed.flatten(-2).gather(-1, selected)
        selected_work -= err[..., None]*u[:, None, None, i, :]
        selected_codes[..., i] = q
        work, codes, costs = selected_work, selected_codes, cost
    return codes[:, :, 0]

_JB_MODE = 'joint'
_JB_MAX_ITEMS = 128
JB_DEBUG = {}
_jb_parent_calibration = hif4_calibration_attention
_jb_parent_q = hif4_dynamic_quantize_q
_jb_parent_queries = _joint_quant_queries


def _jb_mv(x, h):
    return torch.bmm(x[:, None], h).squeeze(1)


def _jb_prepare(st, kh, d):
    h = st['H'].double()
    h = (h+h.transpose(-1, -2))*.5
    h = h/h.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-20)[:, None, None]
    h = h + .01*torch.eye(d, dtype=h.dtype)
    blocks = torch.stack([h[:, i:i+64, i:i+64] for i in range(0, d, 64)], 1)
    chol, info = torch.linalg.cholesky_ex(blocks)
    if bool(info.any()) or not bool(torch.isfinite(h).all()):
        return None
    inv64 = torch.cholesky_solve(torch.eye(64, dtype=h.dtype).expand_as(blocks), chol)
    permutations = []
    for block in blocks.reshape(-1, 64, 64):
        diag = block.diagonal().clamp_min(1e-20).sqrt()
        corr = (block/(diag[:, None]*diag[None])).square()
        remaining = torch.ones(64, dtype=torch.bool)
        order = []
        while len(order)<64:
            first = int(block.diagonal().masked_fill(~remaining, -1).argmax())
            group = [first]; remaining[first] = False
            while len(group)<8:
                next_id = int(corr[group].sum(0).masked_fill(~remaining, -1).argmax())
                group.append(next_id); remaining[next_id] = False
            order.extend(group)
        permutations.append(torch.tensor(order))
    perm = torch.stack(permutations).reshape(*blocks.shape[:2], 64)
    hp = blocks.gather(-2, perm[..., :, None].expand_as(blocks)).gather(-1, perm[..., None, :].expand_as(blocks))
    h8 = torch.stack([hp[..., i:i+8, i:i+8] for i in range(0, 64, 8)], 2)
    ch8, info = torch.linalg.cholesky_ex(h8)
    if bool(info.any()):
        return None
    inv8 = torch.cholesky_solve(torch.eye(8, dtype=h.dtype).expand_as(h8), ch8)
    u8 = torch.linalg.cholesky(inv8, upper=True)
    return dict(st, jb_h=h.float().contiguous(), jb_block=blocks.float().contiguous(),
                jb_inv64=inv64.float().contiguous(), jb_perm=perm.contiguous(),
                jb_hp=hp.float().contiguous(), jb_h8=h8.float().contiguous(),
                jb_inv8=inv8.float().contiguous(), jb_u8=u8.float().contiguous(),
                jb_active=torch.ones(kh, dtype=torch.bool))


def _jb_tree(target, h):
    # Per-item weights: changing another head's active mask cannot renormalize them.
    w = h.diagonal(dim1=-2, dim2=-1).clamp_min(1e-20)
    w = (w/w.mean(-1, keepdim=True)).clamp(.25, 4.) if USE_WEIGHTS else torch.ones_like(w)
    return _quant_chunk(target.reshape(-1, 1, 8, 2, 4), w.reshape(-1, 1, 8, 2, 4), CAND_GRID)


def _jb_code_candidates(tree, target, perm, hp, h8, inv8, u8):
    z = _deq_params(tree).float().gather(1, perm)
    center = target.gather(1, perm)
    step = (_params_unit_flat(tree)*.25).gather(1, perm)
    gradient = _jb_mv(z-center, hp)
    zb, gb = z.reshape(-1, 8, 8), gradient.reshape(-1, 8, 8)
    conditional = zb-torch.matmul(gb.unsqueeze(-2), inv8).squeeze(-2)
    proposed = _jb_beam(conditional.reshape(-1, 1, 8), step.reshape(-1, 1, 8), u8.reshape(-1, 8, 8), 4).reshape_as(zb)
    delta = proposed-zb
    changes = (2*delta*gb+torch.matmul(delta.unsqueeze(-2), h8).squeeze(-2)*delta).sum(-1)
    delta = torch.where((changes<0)[..., None], delta, 0.)
    best_group = changes.argmin(-1)
    single = torch.zeros_like(delta)
    rows = torch.arange(z.shape[0])
    single[rows, best_group] = delta[rows, best_group]
    # Re-evaluate both patterns against the full block outside this helper.
    answers = []
    for update in (delta, single):
        candidate = torch.empty_like(z).scatter_(1, perm, z+update.reshape_as(z))
        answers.append(_values_to_params(candidate, tree))
    return answers


def _jb_propose(parent, target, gradient, h, perm, hp, h8, inv8, u8, mode):
    original = _deq_params(parent).float()
    best, cost = parent, torch.zeros(original.shape[0])
    trees = []
    if mode in ('code', 'joint'):
        trees.append(parent)
    if mode in ('scale', 'joint'):
        trees.append(_jb_tree(target, h))
    for tree in trees:
        candidates = [tree]
        if mode in ('code', 'joint'):
            candidates += _jb_code_candidates(tree, target, perm, hp, h8, inv8, u8)
        for candidate in candidates:
            delta = _deq_params(candidate).float()-original
            newcost = (2*delta*gradient+_jb_mv(delta, h)*delta).sum(-1)
            take = (newcost < cost) & torch.isfinite(newcost)
            mask = take[:, None, None, None, None]
            best = {key: torch.where(mask, candidate[key], best[key]) for key in parent}
            cost = torch.where(take, newcost, cost)
    return best, cost


def _jb_search(x, base, st, qh, d, mode=None):
    t = x.shape[0]
    if not 2<=t<=4096 or d>256 or d%64 or qh>128 or not bool(st['jb_active'].any()):
        return base
    mode = _JB_MODE if mode is None else mode
    groups = torch.arange(qh)*st['jb_h'].shape[0]//qh
    h = st['jb_h'][groups]
    raw = x.reshape(t, qh, d).transpose(0, 1)
    y = _deq_params(base).float().reshape(t, qh, d).transpose(0, 1)
    gradient = torch.bmm(y-raw, h)
    diag = h.diagonal(dim1=-2, dim2=-1).clamp_min(1e-20)
    potential = (gradient.square()/diag[:, None]).reshape(qh, t, d//64, 64).sum(-1)
    score, block = potential.max(-1)
    score = score/(raw.square()*diag[:, None]).sum(-1).clamp_min(1e-20)
    quota = min(max(1, t//8), max(1, _JB_MAX_ITEMS//qh))
    chosen = score.topk(quota, dim=1).indices
    heads = torch.arange(qh)[:, None].expand(qh, quota)
    active = st['jb_active'].repeat_interleave(qh//st['jb_active'].numel())
    rid, hid = chosen[active].reshape(-1), heads[active].reshape(-1)
    bid = block[hid, rid]
    group = groups[hid]
    columns = bid[:, None]*64+torch.arange(64)[None]
    g = gradient[hid, rid].gather(1, columns)
    old = y[hid, rid].gather(1, columns)
    target = old-_jb_mv(g, st['jb_inv64'][group, bid])
    block_ids = hid*(d//64)+bid
    params = {key: value[rid, block_ids].unsqueeze(1) for key, value in base.items()}
    proposal, change = _jb_propose(params, target, g, st['jb_block'][group, bid],
        st['jb_perm'][group, bid], st['jb_hp'][group, bid], st['jb_h8'][group, bid],
        st['jb_inv8'][group, bid], st['jb_u8'][group, bid], mode)
    take = change < 0
    if not bool(take.any()):
        return base
    answer = {key: value.clone() for key, value in base.items()}
    for key in answer:
        answer[key][rid[take], block_ids[take]] = proposal[key][take, 0]
    return answer


def _jb_guard(samples, parent, qh, kh, d):
    JB_DEBUG.clear(); JB_DEBUG.update(accepted=0, changed_queries=0, error=False, mode=_JB_MODE)
    st = parent['q_state']
    if not samples or d%64 or d>256 or qh>128 or not st.get('gq') or 'H' not in st:
        return parent
    hold = samples[-1]; t = hold['q'][0].shape[0]
    if not 4<=t<=4096:
        return parent
    trial = _jb_prepare(st, kh, d)
    if trial is None:
        return parent
    rawq = dequantize_nvfp4(*hold['q']).float()
    base = _jb_parent_q(*hold['q'], qh, d, st)
    proposal = _jb_search(_kt_input(rawq, st, qh, d, 'q'), base, trial, qh, d)
    before, after = _deq_params(base).float(), _deq_params(proposal).float()
    ix = torch.where((before!=after).any(-1))[0]
    JB_DEBUG['changed_queries'] = ix.numel()
    if ix.numel()<2:
        return parent
    rawk, rawv = [dequantize_nvfp4(*hold[r]).float() for r in 'kv']
    target = _joint_output(rawq[ix], rawk, rawv, qh, kh, d, ix, query_subset=True)
    k = _deq_params(hif4_dynamic_quantize_k(*hold['k'], kh, d, parent['k_state']))
    v = _deq_params(hif4_dynamic_quantize_v(*hold['v'], kh, d, parent['v_state']))
    costs = []
    for q in (before[ix], after[ix]):
        e = (_joint_output(q,k,v,qh,kh,d,ix,query_subset=True)-target).square().mean((-1,-2))
        costs.append(torch.stack((e[::2].mean(0), e[1::2].mean(0))))
    take = (costs[1]<costs[0]*.995).all(0) & torch.isfinite(costs[1]).all(0)
    JB_DEBUG.update(accepted=int(take.sum()), ratios=(costs[1].mean(0)/costs[0].mean(0).clamp_min(1e-30)).tolist())
    if not bool(take.any()):
        return parent
    trial['jb_active'] = take
    return dict(parent, q_state=trial)


def hif4_calibration_attention(samples, qh, kh, d):
    parent = _jb_parent_calibration(samples, qh, kh, d)
    try:
        return _jb_guard(samples, parent, qh, kh, d)
    except (RuntimeError, ValueError) as exc:
        JB_DEBUG.update(error=True, error_type=type(exc).__name__)
        return parent


def hif4_dynamic_quantize_q(q_quant, q_scale, q_num_heads, head_dim, q_state):
    base = _jb_parent_q(q_quant, q_scale, q_num_heads, head_dim, q_state)
    if not isinstance(q_state, dict) or 'jb_active' not in q_state:
        return base
    raw = dequantize_nvfp4(q_quant, q_scale).float()
    return _jb_search(_kt_input(raw, q_state, q_num_heads, head_dim, 'q'), base, q_state, q_num_heads, head_dim)


def _joint_quant_queries(hold, state, qh, kh, d, ix):
    if 'jb_active' not in state:
        return _jb_parent_queries(hold, state, qh, kh, d, ix)
    return _deq_params(hif4_dynamic_quantize_q(*hold['q'], qh, d, state))[ix]
