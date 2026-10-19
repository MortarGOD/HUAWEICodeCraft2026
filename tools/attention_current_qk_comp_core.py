"""Current-input blockwise Q-error compensation applied to translated K."""

QKC_DEBUG = {"q_calls": 0, "comp_calls": 0, "fallback_calls": 0}
_QKC_CACHE = {}
_QKC_IN_CALIBRATION = False
_QKC_CAPTURE_Q = False
_QKC_MATRIX = None
_QKC_BLOCK = 1
_QKC_BLEND = 0.5
_QKC_RIDGE = 0.05
_QKC_PARENT_CALIBRATION = hif4_calibration_attention
_QKC_PARENT_Q = hif4_dynamic_quantize_q
_QKC_PARENT_DYN_TABLE = _dyn_table
_QKC_PARENT_KT_INPUT = _kt_input
_QKC_PARENT_TS_QUANT = _ts_quant


def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    global _QKC_IN_CALIBRATION
    _QKC_IN_CALIBRATION = True
    _QKC_CACHE.clear()
    try:
        return _QKC_PARENT_CALIBRATION(calib_qkv_list, q_num_heads, kv_num_heads, head_dim)
    finally:
        _QKC_CACHE.clear()
        _QKC_IN_CALIBRATION = False


def _dyn_table(x, state, has_scale):
    if _QKC_CAPTURE_Q and not _QKC_IN_CALIBRATION and "continuous_q" not in _QKC_CACHE:
        _QKC_CACHE["continuous_q"] = x
    return _QKC_PARENT_DYN_TABLE(x, state, has_scale)


def hif4_dynamic_quantize_q(q_quant, q_scale, q_num_heads, head_dim, q_state):
    global _QKC_CAPTURE_Q
    if _QKC_IN_CALIBRATION:
        return _QKC_PARENT_Q(q_quant, q_scale, q_num_heads, head_dim, q_state)
    _QKC_CACHE.clear()
    _QKC_CACHE.update(qh=int(q_num_heads), d=int(head_dim))
    _QKC_CAPTURE_Q = True
    try:
        result = _QKC_PARENT_Q(q_quant, q_scale, q_num_heads, head_dim, q_state)
    finally:
        _QKC_CAPTURE_Q = False
    _QKC_CACHE["quant_q"] = _deq_params(result)
    QKC_DEBUG["q_calls"] += 1
    return result


def _qkc_fit(cache, kh, d):
    qh = cache["qh"]
    group = qh // kh
    block = _QKC_BLOCK
    quant = cache["quant_q"].reshape(-1, kh, group, d).permute(1, 0, 2, 3).reshape(kh, -1, d)
    target = cache["continuous_q"].reshape(-1, kh, group, d).permute(1, 0, 2, 3).reshape(kh, -1, d)
    quant = quant.reshape(kh, -1, d // block, block).permute(0, 2, 1, 3).float()
    target = target.reshape(kh, -1, d // block, block).permute(0, 2, 1, 3).float()
    gram = quant.transpose(-1, -2) @ quant
    cross = quant.transpose(-1, -2) @ target
    diagonal = gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-8)
    eye = torch.eye(block)[None, None]
    matrix = torch.linalg.solve(gram + (_QKC_RIDGE * diagonal)[..., None, None] * eye, cross)
    matrix = eye + _QKC_BLEND * (matrix - eye)
    if block == 1:
        # Preserve each head's learned attention temperature; compensate only
        # relative channel error, since a uniform K rescale is not softmax-invariant.
        matrix = matrix / matrix.mean(1, keepdim=True).clamp_min(1e-8)
    return matrix


def _kt_input(raw, state, nh, d, role):
    value = _QKC_PARENT_KT_INPUT(raw, state, nh, d, role)
    matrix = _QKC_MATRIX
    if role != "k" or matrix is None or nh != matrix.shape[0] or d % _QKC_BLOCK:
        return value
    block = _QKC_BLOCK
    shaped = value.reshape(-1, nh, d // block, block).permute(1, 2, 0, 3).float()
    shaped = shaped @ matrix.transpose(-1, -2)
    return shaped.permute(2, 0, 1, 3).reshape(value.shape).contiguous()


def _ts_quant(raw, st, kh, d, variant):
    global _QKC_MATRIX
    cache = _QKC_CACHE
    if (
        not _QKC_IN_CALIBRATION
        and "continuous_q" in cache
        and "quant_q" in cache
        and cache.get("d") == int(d)
        and cache.get("qh", 0) % int(kh) == 0
        and d % _QKC_BLOCK == 0
    ):
        try:
            _QKC_MATRIX = _qkc_fit(cache, kh, d)
            QKC_DEBUG["comp_calls"] += 1
            return _QKC_PARENT_TS_QUANT(raw, st, kh, d, variant)
        except Exception:
            pass
        finally:
            _QKC_MATRIX = None
    QKC_DEBUG["fallback_calls"] += 1
    return _QKC_PARENT_TS_QUANT(raw, st, kh, d, variant)
