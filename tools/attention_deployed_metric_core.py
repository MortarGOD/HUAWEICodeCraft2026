"""Calibration Jacobian at actual quantized operands; retain parent shrinkage."""
DM_DEBUG = {}


def _dm_query_gram(query, key, value, d):
    p = torch.softmax(query @ key.T / d**.5, -1)
    out = p @ value
    hq = torch.zeros(d, d)
    for i in range(query.shape[0]):
        jac = key.T @ (p[i, :, None] * (value - out[i])) / d**.5
        hq += jac @ jac.T
    return hq


def _oj_fit(samples, states, qh, kh, d):
    DM_DEBUG.clear()
    DM_DEBUG.update(folds=0, finite=True, point='quantized-QKV')
    folds = []
    for sample in samples[:-1][-2:]:
        n = sample['k'][0].shape[0]
        ix = torch.linspace(0, n-1, min(n, 16)).long()
        # The query helper retains the full sequence's refinement budget.
        query = _joint_quant_queries(sample, states['q_state'], qh, kh, d, ix)
        key = _deq_params(hif4_dynamic_quantize_k(*sample['k'], kh, d, states['k_state']))
        value = _deq_params(hif4_dynamic_quantize_v(*sample['v'], kh, d, states['v_state']))
        query = query.float().reshape(-1, kh, qh//kh, d)
        key = key.float().reshape(n, kh, d)
        value = value.float().reshape(n, kh, d)
        hq = torch.stack([_dm_query_gram(query[:, h].reshape(-1, d), key[:, h], value[:, h], d)
                          for h in range(kh)])
        if not bool(torch.isfinite(hq).all()):
            DM_DEBUG['finite'] = False
            return None
        folds.append(_oj_normalize(hq))
        DM_DEBUG['folds'] += 1
    if not folds:
        return None
    # This variant deliberately quantizes Q only, as does its v382 parent.
    metric = torch.stack(folds).mean(0)
    return metric, torch.zeros_like(metric)
