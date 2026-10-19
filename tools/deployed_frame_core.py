"""Calibration-only complete GPTQ coordinate selection; frozen V and Linear."""
_FRAME_QUERIES = 64
_FRAME_MARGIN = 0.005
FRAME_DEBUG = {}
_frame_parent_calibration = hif4_calibration_attention
_frame_parent_q = hif4_dynamic_quantize_q
_frame_parent_k = hif4_dynamic_quantize_k


def _frame_raw_grams(samples, qh, kh, d):
    hq = torch.zeros(qh, d, d)
    hk = torch.zeros(kh, d, d)
    for sample in samples[:-1]:
        rows = sample['q'][0].shape[0]
        if not 1 <= rows <= 1024:
            continue
        q = dequantize_nvfp4(*sample['q']).float().reshape(-1, qh, d).transpose(0, 1)
        k = dequantize_nvfp4(*sample['k']).float().reshape(-1, kh, d).transpose(0, 1)
        hq += q.transpose(1, 2) @ q
        hk += k.transpose(1, 2) @ k
    return hq, hk


def _frame_candidates(samples, parent, qh, kh, d):
    frames = [parent]
    labels = ['parent']
    rotation = _make_R(d, 0)
    if rotation is None:
        return frames, labels
    rawq, rawk = _frame_raw_grams(samples, qh, kh, d)
    gain = parent['q_state'].get('temperature_gain', torch.ones(qh)).float()
    # K is shared within each GQA group: include every deployed Q gain squared.
    hq = (rawq * gain[:, None, None].square()).reshape(kh, qh // kh, d, d).sum(1)
    options = [('plain_rotation', rotation.expand(kh, -1, -1), rotation.expand(kh, -1, -1))]
    if ('qs' in parent['q_state'] and 'joint_transform' in parent['q_state']
            and 'joint_transform' in parent['k_state']):
        options.append(('keep_dense', parent['q_state']['joint_transform'].float(),
                        parent['k_state']['joint_transform'].float()))
    for label, bq, bk in options:
        frame = {}
        valid = True
        for role, basis, h in (('q', bq, bk.transpose(1, 2) @ rawk @ bk),
                                ('k', bk, bq.transpose(1, 2) @ hq @ bq)):
            h = (h + h.transpose(1, 2)) * 0.5
            if not bool(torch.isfinite(h).all() and (h.diagonal(dim1=-2, dim2=-1).sum(-1)>0).all()):
                valid = False
                break
            u = _upper_cholesky_inv(h)
            if u is None or not bool(torch.isfinite(u).all()):
                valid = False
                break
            st = dict(kvh=kh, rot=1, rseed=0, gq=1,
                      rf=int(parent[role+'_state'].get('rf') or 0),
                      u=u.contiguous(), H=h.bfloat16().contiguous())
            if label == 'keep_dense':
                st['joint_transform'] = basis.contiguous()
            if role == 'q' and 'temperature_gain' in parent['q_state']:
                st['temperature_gain'] = gain.contiguous()
            frame[role+'_state'] = st
        if valid:
            frames.append(frame)
            labels.append(label)
    return frames, labels


def _frame_select(costs, kh):
    qh = costs.shape[-1]
    rep = qh // kh
    choice = torch.zeros(kh, dtype=torch.int64)
    for hv in range(kh):
        sl = slice(hv*rep, (hv+1)*rep)
        best = costs[0, :, sl].sum(-1)
        for index in range(1, costs.shape[0]):
            candidate = costs[index, :, sl].sum(-1)
            if bool(torch.isfinite(candidate).all() and
                    (candidate < best*(1-_FRAME_MARGIN)).all()):
                choice[hv] = index
                best = candidate
    return choice


def _frame_guard(samples, parent, qh, kh, d, context=None):
    frames, labels = _frame_candidates(samples, parent, qh, kh, d)
    if len(frames) == 1:
        return parent
    hold = samples[-1]
    rows = hold['q'][0].shape[0]
    if context:
        ix, target, v = context['ix'], context['target'], context['v']
    else:
        ix = torch.linspace(0, rows-1, min(rows, _FRAME_QUERIES)).long()
        raw = [dequantize_nvfp4(*hold[r]).float() for r in 'qkv']
        target = _joint_output(*raw, qh, kh, d, ix).reshape(-1, qh, d)
        v = _deq_params(hif4_dynamic_quantize_v(*hold['v'], kh, d, parent['v_state']))
    costs = []
    for index, frame in enumerate(frames):
        if index == 0 and context:
            q, k = context['q'], context['k']
        else:
            qs = frame['q_state']
            q = (_joint_quant_queries(hold, qs, qh, kh, d, ix) if qs.get('gq') == 1 else
                 _deq_params(_frame_parent_q(*hold['q'], qh, d, qs))[ix])
            k = _deq_params(_frame_parent_k(*hold['k'], kh, d, frame['k_state']))
        out = _joint_output(q, k, v, qh, kh, d, ix, query_subset=True).reshape(-1, qh, d)
        error = (out-target).square().mean(-1)
        costs.append(torch.stack((error[::2].mean(0), error[1::2].mean(0))))
    costs = torch.stack(costs)
    choice = _frame_select(costs, kh)
    FRAME_DEBUG.update(labels=labels, choice=choice.tolist(), costs=costs.tolist())
    if not bool(choice.any()):
        return parent
    # Keep only selected frames plus the exact parent for the T=1 fallback.
    indices = sorted(set([0]+choice.tolist()))
    mapped = torch.zeros_like(choice)
    for new, old in enumerate(indices):
        mapped[choice == old] = new
    answer = dict(parent)
    for role, rep in (('q', qh//kh), ('k', 1)):
        answer[role+'_state'] = dict(frame_bank=[frames[i][role+'_state'] for i in indices],
                                     frame_choice=mapped.repeat_interleave(rep))
    return answer


def _frame_dynamic(quant, scale, heads, d, state, parent):
    if not isinstance(state, dict) or 'frame_bank' not in state:
        return parent(quant, scale, heads, d, state)
    bank, choice = state['frame_bank'], state['frame_choice']
    if quant.shape[0] <= 1:
        return parent(quant, scale, heads, d, bank[0])
    indices = choice.unique().tolist()
    out = parent(quant, scale, heads, d, bank[indices[0]])
    for index in indices[1:]:
        other = parent(quant, scale, heads, d, bank[index])
        mask = (choice == index).repeat_interleave(d//64)[None, :, None, None, None]
        out = {key: torch.where(mask, other[key], out[key]) for key in out}
    return out


def hif4_dynamic_quantize_q(quant, scale, num_heads, head_dim, q_state):
    return _frame_dynamic(quant, scale, num_heads, head_dim, q_state, _frame_parent_q)


def hif4_dynamic_quantize_k(quant, scale, num_heads, head_dim, k_state):
    return _frame_dynamic(quant, scale, num_heads, head_dim, k_state, _frame_parent_k)


def _frame_calibrate_parent(samples, qh, kh, d):
    # Identical v315 steps, but return ephemeral calibration outputs for reuse.
    # No cross-interface cache or test data is used or stored.
    states = _temp_parent_calibration(samples, qh, kh, d)
    TEMP_DEBUG.clear()
    TEMP_DEBUG.update(heads=0, error=False)
    context = {}
    if len(samples)<2 or d%64 or not states['q_state'].get('gq'):
        return states, context
    if not 4 <= samples[-1]['v'][0].shape[0] <= 4096:
        return states, context
    try:
        delta = _temp_fit(samples, states, qh, kh, d)
        answer = _temp_guard(samples[-1], states, delta, qh, kh, d, context=context)
        return answer, context
    except Exception:
        TEMP_DEBUG['error'] = True
        return states, {}


def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    states, context = _frame_calibrate_parent(calib_qkv_list, q_num_heads, kv_num_heads, head_dim)
    FRAME_DEBUG.clear()
    FRAME_DEBUG.update(error=False)
    if len(calib_qkv_list) < 2 or head_dim % 64:
        return states
    rows = calib_qkv_list[-1]['q'][0].shape[0]
    if not 4 <= rows <= 4096:
        return states
    try:
        return _frame_guard(calib_qkv_list, states, q_num_heads, kv_num_heads, head_dim, context)
    except Exception:
        FRAME_DEBUG['error'] = True
        return states
