"""Unpool GQA query metrics while retaining group-level output validation."""
_ph_parent_fit = _oj_fit
_ph_parent_state = _oj_state
PH_DEBUG = {}


def _oj_fit(samples, states, qh, kh, d):
    PH_DEBUG.clear()
    PH_DEBUG.update(query_heads=qh, kv_heads=kh, folds=0)
    if qh == kh:
        return _ph_parent_fit(samples, states, qh, kh, d)
    folds = []
    for sample in samples[:-1][-2:]:
        n = sample['k'][0].shape[0]
        ix = torch.linspace(0, n-1, min(n, 16)).long()
        query = _joint_quant_queries(sample, states['q_state'], qh, kh, d, ix).float().reshape(-1, qh, d)
        key = _deq_params(hif4_dynamic_quantize_k(*sample['k'], kh, d, states['k_state'])).float().reshape(n, kh, d)
        value = _deq_params(hif4_dynamic_quantize_v(*sample['v'], kh, d, states['v_state'])).float().reshape(n, kh, d)
        h = torch.stack([_dm_query_gram(query[:, j], key[:, j//(qh//kh)], value[:, j//(qh//kh)], d)
                         for j in range(qh)])
        if not bool(torch.isfinite(h).all()):
            return None
        folds.append(_oj_normalize(h))
        PH_DEBUG['folds'] += 1
    if not folds:
        return None
    return torch.stack(folds).mean(0), torch.zeros(kh, d, d)


def _ph_expand(tensor, heads):
    if tensor.shape[0] == heads:
        return tensor
    if heads % tensor.shape[0]:
        raise ValueError('incompatible query metric head count')
    return tensor.repeat_interleave(heads//tensor.shape[0], 0)


def _oj_state(st, h):
    expanded = dict(st)
    for key in ('H', 'u'):
        if isinstance(st.get(key), torch.Tensor):
            expanded[key] = _ph_expand(st[key], h.shape[0])
    # Parent recursion resolves this override for any nested fallback.
    return _ph_parent_state(expanded, h)


def _oj_merge(old, new, mask):
    answer = dict(old)
    for key in ('H', 'u'):
        if key in new:
            heads = new[key].shape[0]
            expanded_mask = _ph_expand(mask, heads)
            fallback = _ph_expand(old.get(key, new[key]), heads)
            answer[key] = torch.where(expanded_mask[:, None, None], new[key], fallback).contiguous()
    if isinstance(old.get('ts_fallback'), dict):
        answer['ts_fallback'] = _oj_merge(old['ts_fallback'], new['ts_fallback'], mask)
    return answer
