"""Calibration-only asymmetric Q/K reconstruction; statically embeddable.

Fit against the quantized counterpart, then guard actual softmax-PV output.
No test-time cross-interface communication or file access. The correction fuses
into an existing dense transform; an originally unrotated path would add a GEMM.
"""
_ASYM_STRENGTH = 0.5
_ASYM_RIDGE = 0.03
_ASYM_FIT_ROWS = 192
_ASYM_GUARD_QUERIES = 32
ASYM_DEBUG = {}


def _asym_transform(x, st, nh, kh, d, role):
    y = x.reshape(-1, nh, d)
    scale = st.get("qs" if role == "q" else "ks")
    if scale is not None:
        y = y * scale.float().repeat_interleave(nh // kh, 0)[None]
    full = st.get("joint_transform")
    if full is not None:
        matrix = full.float().repeat_interleave(nh // full.shape[0], 0)
        y = (y.transpose(0, 1) @ matrix).transpose(0, 1)
    elif st.get("rot"):
        rotation = _make_R(d, int(st.get("rseed") or 0))
        if rotation is not None:
            y = y @ rotation
    return y.reshape(x.shape).contiguous()


def _asym_quant_rows(sample, st, nh, kh, d, role, indices):
    rows = sample[role][0].shape[0]
    if rows == 1:
        return _deq_params(_dyn_qk(*sample[role], st, nh, d, role=role))[indices]
    x = dequantize_nvfp4(sample[role][0][indices], sample[role][1][indices]).float()
    x = _asym_transform(x, st, nh, kh, d, role)
    p = _dyn_table(x, None, has_scale=False)
    unit = _params_unit_flat(p)
    values = None
    if st.get("gq"):
        u = st["u"].float().repeat_interleave(nh // st["u"].shape[0], 0)
        values = _gptq_quantize_batched(
            x.reshape(-1, nh, d).transpose(0, 1).contiguous(),
            unit.reshape(-1, nh, d).transpose(0, 1).contiguous(), u)
        values = values.transpose(0, 1).reshape(x.shape).contiguous()
    rf = int(st.get("rf") or 0)
    if rf and rows <= ATTN_REFINE_MAX_T and "H" in st:
        if values is None:
            values = _deq_params(p)
        rounds = rf * (32 if rows <= 128 else _REF_ATTN_ROUNDS)
        values = _refine_attn_heads(x, values, unit, st["H"], nh, rf,
                                    total_rounds=rounds)
    return _deq_params(_values_to_params(values, p)) if values is not None else _deq_params(p)


def _asym_fit(samples, states, qh, kh, d):
    gram = {r: torch.zeros(kh, d, d) for r in "qk"}
    cross = {r: torch.zeros(kh, d, d) for r in "qk"}
    count = 0
    for sample in samples[:-1]:
        rows = sample["q"][0].shape[0]
        if rows <= 1:
            continue
        ix = torch.linspace(0, rows - 1, min(rows, _ASYM_FIT_ROWS)).long()
        for role, nh in (("q", qh), ("k", kh)):
            st = states[role + "_state"]
            raw = dequantize_nvfp4(sample[role][0][ix], sample[role][1][ix]).float()
            raw = _asym_transform(raw, st, nh, kh, d, role)
            quantized = _asym_quant_rows(sample, st, nh, kh, d, role, ix)
            # Sum all repeated query heads into their shared KV group.
            a = raw.reshape(-1, kh, nh // kh, d).permute(1, 0, 2, 3).reshape(kh, -1, d)
            b = quantized.reshape(-1, kh, nh // kh, d).permute(1, 0, 2, 3).reshape(kh, -1, d)
            gram[role] += b.transpose(1, 2) @ b
            cross[role] += a.transpose(1, 2) @ b
        count += ix.numel()
    if count == 0:
        return None
    proposed = dict(states)
    eye = torch.eye(d)
    for role, other in (("q", "k"), ("k", "q")):
        g = gram[other]
        g = (g + g.transpose(1, 2)) * 0.5
        ridge = g.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-20) * _ASYM_RIDGE
        chol = torch.linalg.cholesky(g + ridge[:, None, None] * eye)
        # C=(raw.T@quant + lambda I) (quant.T@quant + lambda I)^(-1).
        correction = torch.cholesky_solve(
            (cross[other] + ridge[:, None, None] * eye).transpose(1, 2), chol).transpose(1, 2)
        correction = eye + _ASYM_STRENGTH * (correction - eye)
        if not bool(torch.isfinite(correction).all()):
            return None
        # Ill-conditioned populations must not create aggressive amplification.
        singular = torch.linalg.svdvals(correction)
        ASYM_DEBUG[role + "_singular_min"] = float(singular.min())
        ASYM_DEBUG[role + "_singular_max"] = float(singular.max())
        if not bool(((singular > 0.5) & (singular < 2.0)).all()):
            return None
        u = _upper_cholesky_inv(g)
        if u is None:
            return None
        old = states[role + "_state"]
        base = old.get("joint_transform")
        if base is None:
            base = _make_R(d, int(old.get("rseed") or 0)) if old.get("rot") else eye
        candidate = dict(old, joint_transform=(base @ correction).contiguous(), u=u.contiguous())
        if "H" in old:
            candidate["H"] = g.to(torch.bfloat16).contiguous()
        proposed[role + "_state"] = candidate
    return proposed


def _asym_guard(hold, states, proposed, qh, kh, d):
    rows = hold["v"][0].shape[0]
    ix = torch.linspace(0, rows - 1, min(rows, _ASYM_GUARD_QUERIES)).long()
    raw = [dequantize_nvfp4(*hold[r]).float() for r in "qkv"]
    target = _joint_output(*raw, qh, kh, d, ix)
    value = _deq_params(hif4_dynamic_quantize_v(*hold["v"], kh, d, states["v_state"]))
    qs = [_asym_quant_rows(hold, st["q_state"], qh, kh, d, "q", ix) for st in (states, proposed)]
    ks = [_deq_params(hif4_dynamic_quantize_k(*hold["k"], kh, d, st["k_state"])) for st in (states, proposed)]
    costs = torch.empty(2, 2, 2, kh)
    for qi in range(2):
        for ki in range(2):
            err = (_joint_output(qs[qi], ks[ki], value, qh, kh, d, ix, query_subset=True) - target).square().mean((-1, -2))
            costs[qi, ki] = torch.stack((err[::2].mean(0), err[1::2].mean(0)))
    if not bool(torch.isfinite(costs).all()):
        return states
    ASYM_DEBUG["hold_ratios"] = (costs.mean((2, 3)) / costs[0, 0].mean().clamp_min(1e-30)).tolist()
    masks = {r: torch.zeros(kh, dtype=torch.bool) for r in "qk"}
    for hv in range(kh):
        best = costs[0, 0, :, hv]
        for qi in range(2):
            for ki in range(2):
                cand = costs[qi, ki, :, hv]
                if bool((cand < best * 0.995).all()):
                    masks["q"][hv], masks["k"][hv] = bool(qi), bool(ki)
                    best = cand
    answer = dict(states)
    for role in "qk":
        mask = masks[role]
        if not bool(mask.any()):
            continue
        old = states[role + "_state"]
        candidate = dict(proposed[role + "_state"])
        base = old.get("joint_transform")
        if base is None:
            base = _make_R(d, int(old.get("rseed") or 0)) if old.get("rot") else torch.eye(d)
        candidate["joint_transform"] = torch.where(mask[:, None, None], candidate["joint_transform"], base).contiguous()
        for field in ("u", "H"):
            if field in old:
                candidate[field] = torch.where(mask[:, None, None], candidate[field], old[field]).contiguous()
        answer[role + "_state"] = candidate
    ASYM_DEBUG.update(q_groups=int(masks["q"].sum()), k_groups=int(masks["k"].sum()))
    return answer


_asym_parent_calibration = hif4_calibration_attention


def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    states = _asym_parent_calibration(calib_qkv_list, q_num_heads, kv_num_heads, head_dim)
    ASYM_DEBUG.clear()
    ASYM_DEBUG.update(q_groups=0, k_groups=0, error=False)
    if (len(calib_qkv_list) < 2 or not states["q_state"].get("gq")
            or not states["k_state"].get("gq")):
        return states
    rows = calib_qkv_list[-1]["v"][0].shape[0]
    if rows < 4 or rows > 4096:
        return states
    try:
        proposed = _asym_fit(calib_qkv_list, states, q_num_heads, kv_num_heads, head_dim)
        if proposed is None:
            return states
        return _asym_guard(calib_qkv_list[-1], states, proposed, q_num_heads, kv_num_heads, head_dim)
    except Exception:
        ASYM_DEBUG["error"] = True
        return states
