"""Infer a bounded attention kernel from current V and calibration Q/K pairs.

No current Q/K access: calibration fits a regularized low-rank bilinear model.
Actual held-out PV loss decides whether each KV group may use the model.
"""
_VK_RANK = 64
_VK_QUERIES = 4
_VK_ROUNDS = 8
_VK_MSE_CAP = 1.05
JOINT_DEBUG = {}
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
    if x.shape[0] <= 1 or x.shape[0] > 4096 or state is None or not bool(state["active"].any()):
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


def _joint_matrix_power(h, power):
    e, u = torch.linalg.eigh((h + h.transpose(-1, -2)) * 0.5)
    e = e.clamp_min(e.amax(-1)[:, None].clamp_min(1e-20) * 0.001)
    return (u * e.pow(power)[:, None, :]) @ u.transpose(-1, -2)


def _joint_raw_metric(st):
    if "H" in st:
        # Reuse the carried calibration Gram. Recovering it by inverting U
        # performs two ill-conditioned inverse operations and costs O(d^3).
        # Its BF16 rounding already defines the deployed refinement metric.
        h = st["H"].float()
        return (h + h.transpose(-1, -2)) * 0.5
    hd = torch.cholesky_inverse(st["u"].float(), upper=True)
    ridge = hd.diagonal(dim1=-2, dim2=-1).mean(-1) * (GPTQ_DAMP / (1 + GPTQ_DAMP))
    return (hd + hd.transpose(-1, -2)) * 0.5 - ridge[:, None, None] * torch.eye(hd.shape[-1])


def _joint_balanced_states(states, qh, kh, d):
    hk = _joint_raw_metric(states["q_state"])
    hq = _joint_raw_metric(states["k_state"])
    a = hq / (qh // kh)
    ar = _joint_matrix_power(a, 0.5)
    ai = _joint_matrix_power(a, -0.5)
    # M A M = B balances both channel covariances. Use half strength:
    # T=M^(1/4), not a scalar/diagonal smoothing parameter.
    mm = ai @ _joint_matrix_power(ar @ hk @ ar, 0.5) @ ai
    t = _joint_matrix_power(mm, 0.25)
    e, u = torch.linalg.eigh(t)
    t = (u * e.clamp(0.25, 4.0)[:, None, :]) @ u.transpose(-1, -2)
    ti = torch.linalg.inv(t).transpose(-1, -2)
    gq = ti.transpose(-1, -2) @ hk @ ti
    gk = t.transpose(-1, -2) @ hq @ t
    gq = (gq + gq.transpose(-1, -2)) * 0.5
    gk = (gk + gk.transpose(-1, -2)) * 0.5
    uq, uk = _upper_cholesky_inv(gq), _upper_cholesky_inv(gk)
    if uq is None or uk is None:
        return None
    proposed = dict(states)
    for role, transform, gram, inv in (("q", t, gq, uq), ("k", ti, gk, uk)):
        old = states[role + "_state"]
        # Fuse with the old rotation: no second full matrix multiply at test.
        rot = _make_R(d, int(old.get("rseed") or 0)) if old.get("rot") else None
        full = rot @ transform if rot is not None else transform
        st = dict(old, joint_transform=full.contiguous(), u=inv.contiguous())
        if "H" in old:
            st["H"] = gram.to(torch.bfloat16).contiguous()
        proposed[role + "_state"] = st
    return proposed


def _joint_quant_queries(hold, state, qh, kh, d, ix):
    # Query rows are independent in metadata selection, GPTQ and refinement.
    # Quantize only scored queries, keeping the ORIGINAL sequence's budget.
    x = dequantize_nvfp4(hold["q"][0][ix], hold["q"][1][ix]).float()
    if "qs" in state:
        x = _qks_apply_q(x, state["qs"].float(), qh, kh, d, inv=False)
    if "joint_transform" in state:
        t = state["joint_transform"].float().repeat_interleave(qh // kh, 0)
        x = (x.reshape(-1, qh, d).transpose(0, 1) @ t).transpose(0, 1).reshape(x.shape).contiguous()
    elif state.get("rot"):
        rot = _make_R(d, int(state.get("rseed") or 0))
        x = (x.reshape(-1, qh, d) @ rot).reshape(x.shape).contiguous()
    p = _dyn_table(x, None, has_scale=False)
    unit = _params_unit_flat(p)
    u = state["u"].float().repeat_interleave(qh // state["u"].shape[0], 0)
    values = _gptq_quantize_batched(x.reshape(-1, qh, d).transpose(0, 1).contiguous(),
                                  unit.reshape(-1, qh, d).transpose(0, 1).contiguous(), u)
    values = values.transpose(0, 1).reshape(x.shape).contiguous()
    rows = hold["q"][0].shape[0]
    rf = int(state.get("rf") or 0)
    if rf and rows <= ATTN_REFINE_MAX_T and "H" in state:
        rounds = rf * (32 if rows <= 128 else _REF_ATTN_ROUNDS)
        values = _refine_attn_heads(x, values, unit, state["H"], qh, rf,
                                    total_rounds=rounds)
    return _deq_params(_values_to_params(values, p))


def _joint_output(q, k, v, qh, kh, d, ix, query_subset=False):
    qb = (q if query_subset else q[ix]).reshape(-1, qh, d).transpose(0, 1)
    kb = k.reshape(-1, kh, d).transpose(0, 1).repeat_interleave(qh // kh, 0)
    vb = v.reshape(-1, kh, d).transpose(0, 1).repeat_interleave(qh // kh, 0)
    out = torch.softmax(qb @ kb.transpose(1, 2) / (d ** 0.5), -1) @ vb
    return out.transpose(0, 1).reshape(ix.numel(), kh, qh // kh, d)


_joint_parent_calibration = hif4_calibration_attention


def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    states = _joint_parent_calibration(calib_qkv_list, q_num_heads, kv_num_heads, head_dim)
    JOINT_DEBUG.clear()
    JOINT_DEBUG.update(balance_heads=0, kernel_heads=0, error=False)
    qh, kh, d = q_num_heads, kv_num_heads, head_dim
    if len(calib_qkv_list) < 2 or not states["q_state"].get("gq") or not states["k_state"].get("gq"):
        return states
    hold = calib_qkv_list[-1]
    rows = hold["v"][0].shape[0]
    if rows < 4 or rows > 4096:
        return states
    try:
        balanced = _joint_balanced_states(states, qh, kh, d)
        kernel = _vk_fit(calib_qkv_list, qh, kh, d)
        variants = (states, balanced) if balanced is not None else (states,)
        raw = [dequantize_nvfp4(*hold[r]).float() for r in "qkv"]
        vp0 = _vk_parent_v(*hold["v"], None, kh, d)
        vp1 = _vk_refine(raw[2], vp0, kernel, kh, d)
        values = (_deq_params(vp0), _deq_params(vp1))
        # Every key is retained. Subsample only queries, avoiding another
        # full T-by-T softmax allocation in this calibration guard.
        ix = torch.linspace(0, rows - 1, min(rows, 64)).long()
        target = _joint_output(*raw, qh, kh, d, ix)
        costs = torch.empty(len(variants), 2, 2, kh)
        for bi, st in enumerate(variants):
            q = _joint_quant_queries(hold, st["q_state"], qh, kh, d, ix)
            k = _deq_params(hif4_dynamic_quantize_k(*hold["k"], kh, d, st["k_state"]))
            for vi, v in enumerate(values):
                err = (_joint_output(q, k, v, qh, kh, d, ix, query_subset=True) - target).square().mean((-1, -2))
                costs[bi, vi] = torch.stack((err[::2].mean(0), err[1::2].mean(0)))
        if not bool(torch.isfinite(costs).all()):
            return states
        bmask = torch.zeros(kh, dtype=torch.bool)
        vmask = torch.zeros(kh, dtype=torch.bool)
        for hv in range(kh):
            best = costs[0, 0, :, hv]
            for bi in range(len(variants)):
                for vi in range(2):
                    candidate = costs[bi, vi, :, hv]
                    # Both disjoint query folds must improve. The four
                    # combinations are evaluated jointly, not assumed additive.
                    if bool((candidate < best * 0.99).all()):
                        bmask[hv], vmask[hv] = bool(bi), bool(vi)
                        best = candidate
        answer = dict(states)
        if bool(bmask.any()) and balanced is not None:
            for role in "qk":
                old = states[role + "_state"]
                st = dict(balanced[role + "_state"])
                rot = _make_R(d, int(old.get("rseed") or 0)) if old.get("rot") else None
                base = rot if rot is not None else torch.eye(d)
                st["joint_transform"] = torch.where(bmask[:, None, None], st["joint_transform"], base).contiguous()
                for key in ("u", "H"):
                    if key in old:
                        st[key] = torch.where(bmask[:, None, None], st[key], old[key]).contiguous()
                answer[role + "_state"] = st
        if bool(vmask.any()):
            kernel["active"] = vmask
            answer["v_state"] = kernel
        JOINT_DEBUG.update(balance_heads=int(bmask.sum()), kernel_heads=int(vmask.sum()),
                           hold_base=float(costs[0, 0].mean()))
        return answer
    except Exception:
        JOINT_DEBUG["error"] = True
        return states
