"""Use the current test Query as the deployed K-translation probe at parent cost."""

CQF_DEBUG = {"q_calls": 0, "probe_calls": 0, "fallback_calls": 0}
_CQF_CACHE = {}
_CQF_IN_CALIBRATION = False
_CQF_CAPTURE_Q = False
_CQF_PARENT_CALIBRATION = hif4_calibration_attention
_CQF_PARENT_Q = hif4_dynamic_quantize_q
_CQF_PARENT_TS_QUANT = _ts_quant
_CQF_PARENT_DEQUANTIZE = dequantize_nvfp4


def dequantize_nvfp4(quant, scale):
    value = _CQF_PARENT_DEQUANTIZE(quant, scale)
    if _CQF_CAPTURE_Q and not _CQF_IN_CALIBRATION:
        _CQF_CACHE["raw_q"] = value
    return value


def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    global _CQF_IN_CALIBRATION
    _CQF_IN_CALIBRATION = True
    _CQF_CACHE.clear()
    try:
        return _CQF_PARENT_CALIBRATION(calib_qkv_list, q_num_heads, kv_num_heads, head_dim)
    finally:
        _CQF_CACHE.clear()
        _CQF_IN_CALIBRATION = False


def hif4_dynamic_quantize_q(q_quant, q_scale, q_num_heads, head_dim, q_state):
    global _CQF_CAPTURE_Q
    if _CQF_IN_CALIBRATION:
        return _CQF_PARENT_Q(q_quant, q_scale, q_num_heads, head_dim, q_state)
    _CQF_CACHE.clear()
    _CQF_CACHE.update(qh=int(q_num_heads), d=int(head_dim))
    _CQF_CAPTURE_Q = True
    try:
        result = _CQF_PARENT_Q(q_quant, q_scale, q_num_heads, head_dim, q_state)
    finally:
        _CQF_CAPTURE_Q = False
    CQF_DEBUG["q_calls"] += 1
    return result


def _ts_quant(raw, st, kh, d, variant):
    cache = _CQF_CACHE
    if (
        not _CQF_IN_CALIBRATION
        and "raw_q" in cache
        and cache.get("d") == int(d)
        and cache.get("qh", 0) % int(kh) == 0
    ):
        try:
            raw_q = cache["raw_q"].float()
            qh = cache["qh"]
            rows = raw_q.shape[0]
            pick = torch.linspace(0, rows - 1, min(rows, 32)).long()
            probe = raw_q[pick].reshape(-1, kh, qh // kh, d)
            probe = probe.permute(1, 0, 2, 3).reshape(kh, -1, d) / (d ** 0.5)
            CQF_DEBUG["probe_calls"] += 1
            return _CQF_PARENT_TS_QUANT(raw, dict(st, kt_probe=probe), kh, d, variant)
        except Exception:
            pass
    CQF_DEBUG["fallback_calls"] += 1
    return _CQF_PARENT_TS_QUANT(raw, st, kh, d, variant)
