"""Calibration-only, value-aware Gauss--Newton metrics for isolated Q/K APIs.

This source is embedded statically into the standalone submission. It uses
existing torch quantizers; it does not import or access files at runtime.
"""

_PVGN_BLEND = 0.75
_PVGN_KEYS = 64
_PVGN_QUERIES = 8
_PVGN_RANK = 16
_PVGN_GUARD_ROWS = 256
_PVGN_MARGIN = 0.005
PVGN_DEBUG = {}


def _pvgn_transform(x, st, nh, kvh, d, role):
    x = x.reshape(-1, nh, d)
    scale = st.get("qs" if role == "q" else "ks")
    if scale is not None:
        scale = scale.float().repeat_interleave(nh // kvh, 0)
        x = x * scale.unsqueeze(0)
    if st.get("rot"):
        rotation = _make_R(d, int(st.get("rseed") or 0))
        if rotation is not None:
            x = x @ rotation
    return x.transpose(0, 1).contiguous()


def _pvgn_raw_metric(st):
    # u.T @ u = inverse(H + lambda I). Recover the parent's raw H.
    u = st["u"].float()
    hd = torch.cholesky_inverse(u, upper=True)
    ridge = hd.diagonal(dim1=-2, dim2=-1).mean(-1) * (GPTQ_DAMP / (1.0 + GPTQ_DAMP))
    h = hd - ridge[:, None, None] * torch.eye(hd.shape[-1])
    return (h + h.transpose(-1, -2)) * 0.5


def _pvgn_fit(samples, states, qh, kvh, d):
    gq = torch.zeros(qh, d, d)
    gk = torch.zeros(kvh, d, d)
    rep = qh // kvh
    for sample in samples[:-1]:
        q = dequantize_nvfp4(*sample["q"]).float()
        k = dequantize_nvfp4(*sample["k"]).float()
        v = dequantize_nvfp4(*sample["v"]).float()
        rows = min(q.shape[0], k.shape[0], v.shape[0])
        if rows < 2:
            continue
        ik = torch.linspace(0, rows - 1, min(rows, _PVGN_KEYS)).long()
        iq = torch.linspace(0, rows - 1, min(rows, _PVGN_QUERIES)).long()
        qb = _pvgn_transform(q[iq], states["q_state"], qh, kvh, d, "q")
        kb = _pvgn_transform(k[ik], states["k_state"], kvh, kvh, d, "k")
        vb = v[ik].reshape(-1, kvh, d).transpose(0, 1).contiguous()
        # V's constant mode cancels exactly in the softmax derivative.
        vc = vb - vb.mean(1, keepdim=True)
        eig, basis = torch.linalg.eigh(vc @ vc.transpose(1, 2))
        rank = min(_PVGN_RANK, vc.shape[1])
        projected = basis[:, :, -rank:] * eig[:, -rank:].clamp_min(0).sqrt()[:, None, :]
        for h in range(qh):
            hv = h // rep
            p = torch.softmax(qb[h] @ kb[hv].T / (d ** 0.5), -1)
            ov = p @ projected[hv]
            delta = projected[hv][None, :, :] - ov[:, None, :]
            jac = kb[hv].T.unsqueeze(0) @ (p[:, :, None] * delta)
            jf = jac.permute(1, 0, 2).reshape(d, -1)
            gq[h] += (jf @ jf.T) / (d * p.shape[0])
            # The diagonal K-token block of J.T@J, averaged over tokens.
            # Use full V for this scalar norm, not the rank approximation.
            ofull = p @ vc[hv]
            dist = (vc[hv].square().sum(-1)[None, :]
                    - 2 * (ofull @ vc[hv].T)
                    + ofull.square().sum(-1)[:, None]).clamp_min(0)
            weights = (p.square() * dist).sum(-1)
            gk[hv] += (qb[h].T @ (weights[:, None] * qb[h])) / (d * p.shape[0])
    return gq, gk


def _pvgn_proposal(st, metric, nh):
    raw = _pvgn_raw_metric(st)
    raw = raw.repeat_interleave(nh // raw.shape[0], 0)
    a = raw.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-20)
    b = metric.diagonal(dim1=-2, dim2=-1).sum(-1)
    valid = torch.isfinite(b) & (b > 1e-20)
    normalized = metric * (a / b.clamp_min(1e-20))[:, None, None]
    blend = torch.where(valid[:, None, None],
                        (1 - _PVGN_BLEND) * raw + _PVGN_BLEND * normalized, raw)
    u = _upper_cholesky_inv(blend)
    if u is None:
        return None
    out = dict(st, u=u.contiguous())
    if "H" in st:
        out["H"] = blend.to(torch.bfloat16).contiguous()
    return out


def _pvgn_quant_guard(x, st, nh, kvh, d, role, original_rows):
    xt = _pvgn_transform(x, st, nh, kvh, d, role).transpose(0, 1).reshape(x.shape).contiguous()
    p = _dyn_table(xt, None, has_scale=False)
    unit = _params_unit_flat(p)
    u = st["u"].float()
    u = u.repeat_interleave(nh // u.shape[0], 0)
    result = _gptq_quantize_batched(xt.reshape(-1, nh, d).transpose(0, 1).contiguous(),
                                  unit.reshape(-1, nh, d).transpose(0, 1).contiguous(), u)
    result = result.transpose(0, 1).reshape(x.shape).contiguous()
    if st.get("rf", 0) and original_rows <= ATTN_REFINE_MAX_T and "H" in st:
        # Guard subsets are >=129 rows for long inputs: preserve the deployed
        # short/long iteration budget rather than triggering the short branch.
        result = _refine_attn_heads(xt, result, unit, st["H"], nh, int(st["rf"]))
    return _deq_params(_values_to_params(result, p))


def _pvgn_guard(hold, states, proposed, qh, kvh, d):
    raw = [dequantize_nvfp4(*hold[r]).float() for r in "qkv"]
    rows = raw[0].shape[0]
    if rows < 4:
        return states
    ix = torch.linspace(0, rows - 1, min(rows, _PVGN_GUARD_ROWS)).long()
    q, k, v = [x[ix].contiguous() for x in raw]
    # Match the deployed V feedback on the full held-out sequence, then take
    # the same rows. V feedback is cross-token, unlike Q/K quantization.
    vp = _deq_params(_dyn_v(*hold["v"], states["v_state"], kvh, d))[ix]
    ref = _attention_out(q, k, v, qh, kvh, d).reshape(-1, qh, d)
    qq = [_pvgn_quant_guard(q, st["q_state"], qh, kvh, d, "q", rows) for st in (states, proposed)]
    kk = [_pvgn_quant_guard(k, st["k_state"], kvh, kvh, d, "k", rows) for st in (states, proposed)]
    costs = torch.empty(2, 2, 2, qh)
    for qi in range(2):
        for ki in range(2):
            out = _attention_out(qq[qi], kk[ki], vp, qh, kvh, d).reshape(-1, qh, d)
            error = (out - ref).square().mean(-1)
            costs[qi, ki] = torch.stack([error[::2].mean(0), error[1::2].mean(0)])
    if not bool(torch.isfinite(costs).all()):
        return states
    qmask = torch.zeros(qh, dtype=torch.bool)
    kmask = torch.zeros(kvh, dtype=torch.bool)
    rep = qh // kvh
    for hv in range(kvh):
        sl = slice(hv * rep, (hv + 1) * rep)
        # Under K0, choose Q1 only if it wins on both query folds.
        q0 = (costs[1, 0, :, sl] < (1 - _PVGN_MARGIN) * costs[0, 0, :, sl]).all(0)
        current = torch.where(q0[None], costs[1, 0, :, sl], costs[0, 0, :, sl])
        # K is shared by the GQA group. Its replacement must improve BOTH
        # folds, and cannot sacrifice an individual head by more than 1%.
        q1 = (costs[1, 1, :, sl] < costs[0, 1, :, sl]).all(0)
        cand = torch.where(q1[None], costs[1, 1, :, sl], costs[0, 1, :, sl])
        take_k = bool((cand.sum(-1) < (1 - _PVGN_MARGIN) * current.sum(-1)).all()
                      and (cand <= current * 1.01).all())
        kmask[hv] = take_k
        qmask[sl] = q1 if take_k else q0
    answer = dict(states)
    for role, mask in (("q", qmask), ("k", kmask)):
        parent, candidate = states[role + "_state"], proposed[role + "_state"]
        if not bool(mask.any()):
            continue
        result = dict(parent)
        for field in ("u", "H"):
            if field in parent and field in candidate:
                base = parent[field].repeat_interleave(mask.numel() // parent[field].shape[0], 0)
                result[field] = torch.where(mask[:, None, None], candidate[field], base).contiguous()
        answer[role + "_state"] = result
    PVGN_DEBUG.update(q_heads=int(qmask.sum()), k_heads=int(kmask.sum()),
                      hold_base=float(costs[0, 0].mean()))
    return answer


_pvgn_parent_calibration = hif4_calibration_attention


def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    states = _pvgn_parent_calibration(calib_qkv_list, q_num_heads, kv_num_heads, head_dim)
    PVGN_DEBUG.clear()
    PVGN_DEBUG.update(q_heads=0, k_heads=0, error=False)
    if (len(calib_qkv_list) < 2 or q_num_heads % kv_num_heads
            or not states["q_state"].get("gq") or not states["k_state"].get("gq")):
        return states
    try:
        gq, gk = _pvgn_fit(calib_qkv_list, states, q_num_heads, kv_num_heads, head_dim)
        pq = _pvgn_proposal(states["q_state"], gq, q_num_heads)
        pk = _pvgn_proposal(states["k_state"], gk, kv_num_heads)
        if pq is None or pk is None:
            return states
        proposed = dict(states, q_state=pq, k_state=pk)
        return _pvgn_guard(calib_qkv_list[-1], states, proposed,
                           q_num_heads, kv_num_heads, head_dim)
    except Exception:
        # A missing/degenerate calibration statistic must never break the
        # independently probed contestant interface.
        PVGN_DEBUG["error"] = True
        return states
