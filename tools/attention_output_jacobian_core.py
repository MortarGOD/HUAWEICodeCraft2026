"""Output-Jacobian Gram, independently derived in deployed Q/K coordinates.

Calibration-only fit. No dynamic cross-role cache or test-output access.
Jq = K.T diag(p) (V - pV) / sqrt(d).
K diagonal-token Gauss-Newton = sum p_j^2 ||V_j-pV||^2 qq.T / d.
"""
OJ_ROLE = 'q'
OJ_DEBUG = {}
_oj_parent_calibration = hif4_calibration_attention


def _oj_normalize(h):
    return h / h.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-20)[:, None, None]


def _oj_fit(samples, states, qh, kh, d):
    folds = []
    for sample in samples[:-1][-2:]:
        n = sample['k'][0].shape[0]
        ix = torch.linspace(0, n-1, min(n, 16)).long()
        rawq = dequantize_nvfp4(sample['q'][0][ix], sample['q'][1][ix]).float()
        rawk = dequantize_nvfp4(*sample['k']).float()
        rawv = dequantize_nvfp4(*sample['v']).float()
        q = _kt_input(rawq, states['q_state'], qh, d, 'q').reshape(-1, kh, qh//kh, d)
        k = _kt_input(rawk, states['k_state'], kh, d, 'k').reshape(n, kh, d)
        v = rawv.reshape(n, kh, d)
        hq = torch.zeros(kh, d, d); hk = torch.zeros_like(hq)
        # Work one KV head at a time: peak memory O(sampled_queries * n).
        for h in range(kh):
            query = q[:, h].reshape(-1, d)
            key, value = k[:, h], v[:, h]
            p = torch.softmax(query @ key.T / d**.5, -1)
            out = p @ value
            for i in range(query.shape[0]):
                centered = value-out[i]
                jac = key.T @ (p[i, :, None]*centered) / d**.5
                hq[h] += jac @ jac.T
            sensitivity = (p.square() * (value.square().sum(-1)[None]
                - 2*out @ value.T + out.square().sum(-1)[:, None]).clamp_min(0)).sum(-1)
            hk[h] = query.T @ (query * sensitivity[:, None]) / d
        folds.append((_oj_normalize(hq), _oj_normalize(hk)))
    if not folds:
        return None
    return tuple(torch.stack([f[i] for f in folds]).mean(0) for i in (0, 1))


def _oj_state(st, h):
    # Update all fallback paths in the same transformed coordinate system.
    answer = dict(st)
    for key in ('ts_fallback',):
        if isinstance(st.get(key), dict):
            answer[key] = _oj_state(st[key], h)
    old = st.get('H')
    if not isinstance(old, torch.Tensor):
        u = st.get('u')
        if not isinstance(u, torch.Tensor):
            return answer
        old = torch.linalg.inv(u.float().transpose(-1, -2) @ u.float())
    # Fixed shrinkage preserves logit directions absent in small calibration sets.
    metric = .5*_oj_normalize(old.float()) + .5*h
    upper = _upper_cholesky_inv(metric)
    if upper is not None:
        answer['H'] = metric.cpu().contiguous()
        answer['u'] = upper.cpu().contiguous()
    return answer


def _oj_merge(old, new, mask):
    answer = dict(old)
    for key in ('H', 'u'):
        if key in new:
            fallback = old.get(key, new[key])
            answer[key] = torch.where(mask[:, None, None], new[key], fallback).contiguous()
    if isinstance(old.get('ts_fallback'), dict):
        answer['ts_fallback'] = _oj_merge(old['ts_fallback'], new['ts_fallback'], mask)
    return answer


@torch.inference_mode()
def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    parent = _oj_parent_calibration(calib_qkv_list, q_num_heads, kv_num_heads, head_dim)
    qh, kh, d = q_num_heads, kv_num_heads, head_dim
    OJ_DEBUG.clear(); OJ_DEBUG.update(accepted=0, error=False, role=OJ_ROLE)
    if len(calib_qkv_list)<3 or d%64 or not all(parent[r+'_state'].get('gq') for r in 'qk'):
        return parent
    hold = calib_qkv_list[-1]; n = hold['k'][0].shape[0]
    if not 4<=n<=4096:
        return parent
    try:
        metrics = _oj_fit(calib_qkv_list, parent, qh, kh, d)
        if metrics is None:
            return parent
        trial = dict(parent)
        for i, role in enumerate('qk'):
            if role in OJ_ROLE:
                trial[role+'_state'] = _oj_state(parent[role+'_state'], metrics[i])
        ix = torch.linspace(0, n-1, min(n, 32)).long()
        raw = [dequantize_nvfp4(hold['q'][0][ix], hold['q'][1][ix]).float()]
        raw += [dequantize_nvfp4(*hold[r]).float() for r in 'kv']
        target = _joint_output(*raw, qh, kh, d, ix, query_subset=True)
        value = _deq_params(hif4_dynamic_quantize_v(*hold['v'], kh, d, parent['v_state']))
        def cost(st):
            query = _joint_quant_queries(hold, st['q_state'], qh, kh, d, ix)
            key = _deq_params(hif4_dynamic_quantize_k(*hold['k'], kh, d, st['k_state']))
            err = (_joint_output(query,key,value,qh,kh,d,ix,query_subset=True)-target).square().mean((-1,-2))
            return torch.stack([err[::2].mean(0), err[1::2].mean(0)])
        before, after = cost(parent), cost(trial)
        take = (after < before*.995).all(0) & torch.isfinite(after).all(0)
        OJ_DEBUG.update(accepted=int(take.sum()), ratios=(after.mean(0)/before.mean(0).clamp_min(1e-30)).tolist())
        for role in OJ_ROLE:
            trial[role+'_state'] = _oj_merge(parent[role+'_state'],trial[role+'_state'],take)
        return trial if bool(take.any()) else parent
    except (RuntimeError, ValueError) as exc:
        OJ_DEBUG.update(error=True, error_type=type(exc).__name__)
        return parent
