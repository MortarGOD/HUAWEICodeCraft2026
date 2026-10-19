"""Infer a bounded attention kernel from current V and calibration Q/K pairs.

No current Q/K access: calibration fits a regularized low-rank bilinear model.
Actual held-out PV loss decides whether each KV group may use the model.
"""
_VK_RANK = 32
_VK_QUERIES = 4
_VK_ROUNDS = 8
_VK_MSE_CAP = 1.05
VK_DEBUG = {}
_vk_parent_calibration = hif4_calibration_attention
_vk_parent_v = _dyn_v


def _vk_features(x, state):
    f = (x - state["center"].float()) @ state["projection"].float()
    return torch.cat((f, torch.ones(f.shape[0], 1)), 1)


def _vk_fit(samples, qh, kh, d):
    fitted = {r: [] for r in "qkv"}
    for smp in samples[:-1]:
        t = smp["v"][0].shape[0]
        ix = torch.linspace(0, t - 1, min(256, t)).long()
        for role in "qkv":
            fitted[role].append(dequantize_nvfp4(*smp[role]).float()[ix])
    q, k, v = [torch.cat(fitted[r]) for r in "qkv"]
    center = v.mean(0)
    vc = v - center
    cov = vc.T @ vc / v.shape[0]
    eig, basis = torch.linalg.eigh(cov)
    rank = min(_VK_RANK, v.shape[0] - 1, v.shape[1])
    ridge = eig[-1].clamp_min(1e-12) * 0.01
    proj = basis[:, -rank:] / (eig[-rank:].clamp_min(0) + ridge).sqrt()[None]
    # Round once before fitting so deployment sees exactly these features.
    state = {"center": center.to(torch.bfloat16).contiguous(),
             "projection": proj.to(torch.bfloat16).contiguous()}
    f = _vk_features(v, state)
    h = f.T @ f + torch.eye(f.shape[1]) * (0.03 * f.shape[0])
    chol = torch.linalg.cholesky(h)
    bq = torch.cholesky_solve(f.T @ q, chol).reshape(-1, qh, d).permute(1, 0, 2)
    bk = torch.cholesky_solve(f.T @ k, chol).reshape(-1, kh, d).permute(1, 0, 2)
    bk = bk.repeat_interleave(qh // kh, 0)
    kernel = (bq @ bk.transpose(1, 2)) / (d ** 0.5)
    state.update(kernel=kernel.to(torch.bfloat16).contiguous(),
                 heads=qh, active=torch.ones(kh, dtype=torch.bool))
    return state


def _vk_probabilities(x, state, kh):
    f = _vk_features(x, state)
    ix = torch.linspace(0, x.shape[0] - 1, min(_VK_QUERIES, x.shape[0])).long()
    logits = (f[ix].unsqueeze(0) @ state["kernel"].float()) @ f.T
    p = torch.softmax(logits, -1)
    return p.reshape(kh, -1, x.shape[0]).contiguous()


def _vk_refine(x, p, state, kh, d):
    if x.shape[0] <= 1 or state is None or not bool(state["active"].any()):
        return p
    probs = _vk_probabilities(x, state, kh)
    steps = _params_unit_flat(p) * 0.25
    baseline = _deq_params(p)
    xh = x.reshape(-1, kh, d).transpose(0, 1).contiguous()
    best = baseline.reshape(-1, kh, d).transpose(0, 1).contiguous()
    step = steps.reshape(-1, kh, d).transpose(0, 1).contiguous()
    penalty = 1.0 / x.shape[0]
    error = xh - best
    original_mse = error.square().mean(1).clamp_min(1e-30)
    best_cost = (probs @ error).square().mean(1) + penalty * original_mse
    mass = probs.sum(1).clamp_min(1e-12)[:, :, None]
    shift = torch.zeros_like(best)
    for _ in range(_VK_ROUNDS):
        residual = probs @ error
        gradient = probs.transpose(1, 2) @ residual
        shift = shift + 0.75 * gradient / mass
        proposal = torch.round((xh + shift) / step).clamp(-7, 7) * step
        error = xh - proposal
        mse = error.square().mean(1)
        cost = (probs @ error).square().mean(1) + penalty * mse
        accept = (cost < best_cost) & (mse <= original_mse * _VK_MSE_CAP)
        accept = accept & state["active"][:, None]
        best = torch.where(accept[:, None], proposal, best)
        best_cost = torch.where(accept, cost, best_cost)
    return _values_to_params(best.transpose(0, 1).reshape(x.shape).contiguous(), p)


def _dyn_v(quant, scale, state, kvh, dh):
    p = _vk_parent_v(quant, scale, None, kvh, dh)
    if not isinstance(state, dict) or "kernel" not in state:
        return p
    return _vk_refine(dequantize_nvfp4(quant, scale).float(), p, state, kvh, dh)


def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    states = _vk_parent_calibration(calib_qkv_list, q_num_heads, kv_num_heads, head_dim)
    VK_DEBUG.clear()
    VK_DEBUG.update(accepted=0, error=False)
    if len(calib_qkv_list) < 2:
        return states
    qh, kh, d = q_num_heads, kv_num_heads, head_dim
    try:
        candidate = _vk_fit(calib_qkv_list, qh, kh, d)
        hold = calib_qkv_list[-1]
        raw = [dequantize_nvfp4(*hold[r]).float() for r in "qkv"]
        if raw[0].shape[0] < 4:
            return states
        q = _deq_params(hif4_dynamic_quantize_q(*hold["q"], qh, d, states["q_state"]))
        k = _deq_params(hif4_dynamic_quantize_k(*hold["k"], kh, d, states["k_state"]))
        vp0 = _vk_parent_v(*hold["v"], None, kh, d)
        vp1 = _vk_refine(raw[2], vp0, candidate, kh, d)
        target = _attention_out(*raw, qh, kh, d).reshape(-1, kh, qh // kh, d)
        losses = []
        for p in (vp0, vp1):
            out = _attention_out(q, k, _deq_params(p), qh, kh, d).reshape(target.shape)
            e = (out - target).square().mean((-1, -2))
            losses.append(torch.stack((e[::2].mean(0), e[1::2].mean(0))))
        active = (losses[1] < losses[0] * 0.99).all(0)
        VK_DEBUG.update(accepted=int(active.sum()), base=float(losses[0].mean()),
                        candidate=float(losses[1].mean()))
        if bool(active.any()):
            candidate["active"] = active
            states = dict(states, v_state=candidate)
    except Exception:
        VK_DEBUG["error"] = True
    return states
