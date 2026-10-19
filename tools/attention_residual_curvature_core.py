"""v393: PSD projection of the exact local Q-output loss Hessian."""
EH_DEBUG = {}


def _eh_query_terms(query, key, value, target, d):
    p = torch.softmax(query @ key.T / d**.5, -1)
    out = p @ value
    gn = torch.zeros(d, d, dtype=query.dtype, device=query.device)
    for i in range(query.shape[0]):
        jac = key.T @ (p[i, :, None]*(value-out[i])) / d**.5
        gn += jac @ jac.T
    residual = out-target
    weight = p * (residual @ value.T - (residual*out).sum(-1, keepdim=True))
    mean = p @ key
    weighted_key = weight @ key
    cross = weighted_key.T @ mean
    curvature = (key.T @ (weight.sum(0)[:, None]*key) - cross - cross.T
                 + mean.T @ (weight.sum(-1)[:, None]*mean)) / d
    return gn, (curvature+curvature.T)*.5


def _eh_project(h):
    eigen, axes = torch.linalg.eigh((h+h.T)*.5)
    return (axes*eigen.clamp_min(0)[None]) @ axes.T, int((eigen < 0).sum())


def _oj_fit(samples, states, qh, kh, d):
    EH_DEBUG.clear(); EH_DEBUG.update(folds=0, finite=True, negative=0, relative=[])
    folds = []
    for sample in samples[:-1][-2:]:
        n = sample['k'][0].shape[0]
        ix = torch.linspace(0, n-1, min(n, 16)).long()
        rawq = dequantize_nvfp4(sample['q'][0][ix], sample['q'][1][ix]).float()
        rawk = dequantize_nvfp4(*sample['k']).float()
        rawv = dequantize_nvfp4(*sample['v']).float()
        target = _joint_output(rawq, rawk, rawv, qh, kh, d, ix, query_subset=True)
        query = _joint_quant_queries(sample, states['q_state'], qh, kh, d, ix)
        key = _deq_params(hif4_dynamic_quantize_k(*sample['k'], kh, d, states['k_state']))
        value = _deq_params(hif4_dynamic_quantize_v(*sample['v'], kh, d, states['v_state']))
        query = query.float().reshape(-1, kh, qh//kh, d)
        key = key.float().reshape(n, kh, d)
        value = value.float().reshape(n, kh, d)
        metrics = []
        for head in range(kh):
            gn, curvature = _eh_query_terms(query[:, head].reshape(-1, d), key[:, head],
                                           value[:, head], target[:, head].reshape(-1, d), d)
            if not bool(torch.isfinite(gn+curvature).all()):
                EH_DEBUG['finite'] = False
                return None
            metric, negative = _eh_project(gn+curvature)
            metrics.append(metric)
            EH_DEBUG['negative'] += negative
            EH_DEBUG['relative'].append(float(curvature.norm()/gn.norm().clamp_min(1e-30)))
        folds.append(_oj_normalize(torch.stack(metrics)))
        EH_DEBUG['folds'] += 1
    if not folds:
        return None
    metric = torch.stack(folds).mean(0)
    return metric, torch.zeros_like(metric)
