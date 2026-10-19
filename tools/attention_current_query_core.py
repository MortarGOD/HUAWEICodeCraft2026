"""Test-time current-Query K-origin selection for sequential Attention calls."""

CQ_DEBUG = {"q_calls": 0, "k_calls": 0, "matched": 0, "accepted_heads": 0}
_CQ_CACHE = {}
_CQ_PARENT_Q = hif4_dynamic_quantize_q
_CQ_PARENT_K = hif4_dynamic_quantize_k


def _cq_probability(query, key, qh, kh, d):
    qb = query.reshape(-1, qh, d).transpose(0, 1)
    kb = key.reshape(-1, kh, d).transpose(0, 1).repeat_interleave(qh // kh, 0)
    return torch.softmax(qb @ kb.transpose(1, 2) / (d ** 0.5), -1)


def hif4_dynamic_quantize_q(q_quant, q_scale, q_num_heads, head_dim, q_state):
    result = _CQ_PARENT_Q(q_quant, q_scale, q_num_heads, head_dim, q_state)
    raw = dequantize_nvfp4(q_quant, q_scale).float()
    _CQ_CACHE.clear()
    _CQ_CACHE.update(
        raw_q=raw,
        quant_q=_deq_params(result),
        qh=int(q_num_heads),
        d=int(head_dim),
    )
    CQ_DEBUG["q_calls"] += 1
    return result


def hif4_dynamic_quantize_k(k_quant, k_scale, kv_num_heads, head_dim, k_state):
    parent = _CQ_PARENT_K(k_quant, k_scale, kv_num_heads, head_dim, k_state)
    CQ_DEBUG["k_calls"] += 1
    cache = _CQ_CACHE
    if (
        not isinstance(k_state, dict)
        or "kt_probe" not in k_state
        or "raw_q" not in cache
        or cache.get("d") != int(head_dim)
        or cache.get("qh", 0) % int(kv_num_heads)
        or not 4 <= k_quant.shape[0] <= 4096
    ):
        return parent
    try:
        raw_k = dequantize_nvfp4(k_quant, k_scale).float()
        raw_q = cache["raw_q"]
        quant_q = cache["quant_q"]
        qh, kh, d = cache["qh"], int(kv_num_heads), int(head_dim)
        q_rows = raw_q.shape[0]
        pick = torch.linspace(0, q_rows - 1, min(q_rows, 32)).long()
        probe = raw_q[pick].reshape(-1, kh, qh // kh, d)
        probe = probe.permute(1, 0, 2, 3).reshape(kh, -1, d) / (d ** 0.5)
        candidate_state = dict(k_state, kt_probe=probe, kt_active=torch.ones(kh, dtype=torch.bool))
        candidate = _ts_quant(raw_k, candidate_state, kh, d, 1)
        parent_key = _deq_params(parent)
        candidate_key = _deq_params(candidate)
        target_probability = _cq_probability(raw_q[pick], raw_k, qh, kh, d)
        parent_probability = _cq_probability(quant_q[pick], parent_key, qh, kh, d)
        candidate_probability = _cq_probability(quant_q[pick], candidate_key, qh, kh, d)
        group = qh // kh
        old = (parent_probability - target_probability).square().mean((-1, -2)).reshape(kh, group).mean(1)
        new = (candidate_probability - target_probability).square().mean((-1, -2)).reshape(kh, group).mean(1)
        active = new < old * 0.9975
        CQ_DEBUG["matched"] += 1
        CQ_DEBUG["accepted_heads"] += int(active.sum())
        if not bool(active.any()):
            return parent
        blocks = active.repeat_interleave(d // 64)
        return {
            name: torch.where(
                blocks.reshape(1, -1, *([1] * (value.ndim - 2))),
                value,
                parent[name],
            )
            for name, value in candidate.items()
        }
    except Exception:
        return parent
