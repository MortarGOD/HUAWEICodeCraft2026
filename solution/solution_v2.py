"""Candidate v1: task-aware hierarchical HiF4 quantization.

The public HiF4 representation is fixed, but its hierarchical scales are not
uniquely determined.  This implementation searches a small E6M2 neighbourhood
and exactly minimizes a diagonal quadratic objective over the E1_8/E1_16 tree.
It uses only PyTorch and keeps all calibration state as finite CPU tensors.
"""

from __future__ import annotations

from typing import Any, Iterable

import math

import torch


E6M2_MIN = 2.0 ** -48
E6M2_MAX = 49152.0


def dequantize_nvfp4(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    blk_size: int = 16,
) -> torch.Tensor:
    channels = int(quant_float.shape[-1])
    if channels % blk_size != 0:
        raise ValueError(f"last dimension {channels} is not divisible by {blk_size}")
    values = quant_float.unflatten(-1, (-1, blk_size))
    return (values * scale_float.unsqueeze(-1)).flatten(-2, -1).to(torch.bfloat16)


def _e6m2_ceil_index(scale: torch.Tensor) -> torch.Tensor:
    """Return the monotonic index of the first legal E6M2 value >= scale."""
    safe = scale.float().clamp(min=E6M2_MIN, max=E6M2_MAX)
    exponent = torch.floor(torch.log2(safe))
    normalized = safe * torch.pow(2.0, -exponent)
    mantissa = torch.ceil((normalized - 1.0) * 4.0)
    carry = mantissa >= 4.0
    exponent = exponent + carry.to(exponent.dtype)
    mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa)
    index = ((exponent + 48.0) * 4.0 + mantissa).to(torch.int64)
    return index.clamp_(0, 254)


def _e6m2_from_index(index: torch.Tensor) -> torch.Tensor:
    index = index.clamp(0, 254)
    exponent = torch.div(index, 4, rounding_mode="floor").float() - 48.0
    mantissa = torch.remainder(index, 4).float()
    value = torch.pow(2.0, exponent) * (1.0 + mantissa * 0.25)
    return value.clamp_(E6M2_MIN, E6M2_MAX)


def _scaled_error(
    grouped: torch.Tensor,
    scale: torch.Tensor,
    multiplier: float,
    importance: torch.Tensor | None,
) -> torch.Tensor:
    local_scale = scale[..., None, None, None] * multiplier
    mant = torch.round(grouped.abs() / local_scale * 4.0) * 0.25
    mant = mant.clamp_(0.0, 1.75)
    reconstructed = torch.sign(grouped) * mant * local_scale
    error = (reconstructed - grouped).square_()
    if importance is not None:
        error.mul_(importance)
    return error.sum(dim=-1)


def _tree_error(
    grouped: torch.Tensor,
    scale: torch.Tensor,
    importance: torch.Tensor | None,
) -> torch.Tensor:
    err1 = _scaled_error(grouped, scale, 1.0, importance)
    err2 = _scaled_error(grouped, scale, 2.0, importance)
    err4 = _scaled_error(grouped, scale, 4.0, importance)
    level2_one = torch.minimum(err1, err2).sum(dim=-1)
    level2_two = torch.minimum(err2, err4).sum(dim=-1)
    return torch.minimum(level2_one, level2_two).sum(dim=-1)


def _materialize_tree(
    grouped: torch.Tensor,
    scale: torch.Tensor,
    importance: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    err1 = _scaled_error(grouped, scale, 1.0, importance)
    err2 = _scaled_error(grouped, scale, 2.0, importance)
    err4 = _scaled_error(grouped, scale, 4.0, importance)

    level2_one = torch.minimum(err1, err2).sum(dim=-1)
    level2_two = torch.minimum(err2, err4).sum(dim=-1)
    use_level2_two = level2_two < level2_one
    level2 = torch.where(use_level2_two, 2.0, 1.0)

    use_level3_when_one = err2 < err1
    use_level3_when_two = err4 < err2
    use_level3 = torch.where(
        use_level2_two.unsqueeze(-1),
        use_level3_when_two,
        use_level3_when_one,
    )
    level3 = torch.where(use_level3, 2.0, 1.0)
    total_scale = scale[..., None, None, None] * level2[..., None, None] * level3[..., None]
    mant = (torch.round(grouped.abs() / total_scale * 4.0) * 0.25).clamp_(0.0, 1.75)
    sign = torch.where(mant == 0.0, 0.0, torch.sign(grouped))
    return level2, level3, sign, mant


def _normalize_importance(
    importance: torch.Tensor | None,
    x: torch.Tensor,
) -> torch.Tensor | None:
    if importance is None:
        return None
    weights = importance.detach().to(device=x.device, dtype=torch.float32)
    if weights.ndim != 1 or int(weights.numel()) != int(x.shape[-1]):
        return None
    weights = weights.clamp_min(0.0)
    weights = weights / weights.mean().clamp_min(1.0e-12)
    # A uniform component limits calibration overfitting and protects unseen data.
    weights = 0.25 + 0.75 * weights.sqrt()
    prefix = (1,) * (x.ndim - 1)
    return weights.reshape(*prefix, int(x.shape[-1])).unflatten(-1, (-1, 64)).reshape(
        *prefix, int(x.shape[-1]) // 64, 8, 2, 4
    )


def _quantize_hif4_search(
    x: torch.Tensor,
    importance: torch.Tensor | None = None,
    offsets: Iterable[int] = (-2, -1, 0),
) -> dict[str, torch.Tensor]:
    x = x.float()
    shape = tuple(int(dim) for dim in x.shape)
    channels = shape[-1]
    if channels % 64 != 0:
        raise ValueError(f"last dimension {channels} is not divisible by 64")
    blocks = channels // 64
    grouped = x.unflatten(-1, (-1, 64)).reshape(*shape[:-1], blocks, 8, 2, 4)
    grouped_importance = _normalize_importance(importance, x)

    peak = grouped.abs().amax(dim=(-1, -2, -3))
    base_index = _e6m2_ceil_index(peak * (1.0 / 7.0))
    best_error = torch.full_like(peak, torch.inf)
    best_index = base_index
    for offset in offsets:
        candidate_index = (base_index + int(offset)).clamp(0, 254)
        candidate_scale = _e6m2_from_index(candidate_index)
        candidate_error = _tree_error(grouped, candidate_scale, grouped_importance)
        improve = candidate_error < best_error
        best_error = torch.where(improve, candidate_error, best_error)
        best_index = torch.where(improve, candidate_index, best_index)

    scale = _e6m2_from_index(best_index)
    level2, level3, sign, mant = _materialize_tree(grouped, scale, grouped_importance)
    return {
        "scale_factor": scale.reshape(*shape[:-1], blocks, 1, 1, 1),
        "scale_lv2": level2.reshape(*shape[:-1], blocks, 8, 1, 1),
        "scale_lv3": level3.reshape(*shape[:-1], blocks, 8, 2, 1),
        "sign": sign.reshape(*shape[:-1], blocks, 8, 2, 4),
        "mant": mant.reshape(*shape[:-1], blocks, 8, 2, 4),
    }


def _second_moment_from_pairs(pairs: list) -> torch.Tensor:
    total: torch.Tensor | None = None
    count = 0
    for quant, scale in pairs:
        values = dequantize_nvfp4(quant, scale).float()
        moment = values.square().sum(dim=tuple(range(values.ndim - 1)))
        total = moment if total is None else total + moment
        count += values.numel() // values.shape[-1]
    assert total is not None
    return (total / max(count, 1)).detach().cpu().float().contiguous()


def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    weight = dequantize_nvfp4(weight_quant, weight_scale).float()
    weight_peak = weight.abs().amax(dim=tuple(range(weight.ndim - 1))).clamp_min_(1.0e-8)
    activation_peak: torch.Tensor | None = None
    for quant, scale in calib_activation_list:
        values = dequantize_nvfp4(quant, scale).float()
        peak = values.abs().amax(dim=tuple(range(values.ndim - 1)))
        activation_peak = peak if activation_peak is None else torch.maximum(activation_peak, peak)
    assert activation_peak is not None
    activation_peak.clamp_min_(1.0e-8)

    # Diagonal SmoothQuant transform.  In exact arithmetic
    # (A / smooth) @ (W * smooth)^T == A @ W^T.
    alpha = 0.875
    smooth = activation_peak.pow(alpha) / weight_peak.pow(1.0 - alpha)
    smooth = smooth / smooth.log().mean().exp()
    smooth.clamp_(1.0 / 16.0, 16.0)
    weight_params = _quantize_hif4_search(
        weight * smooth,
        importance=None,
        offsets=(-1, 0, 1, 2, 3, 4),
    )
    return {
        "weight_params": weight_params,
        "activation_state": {"smooth": smooth.detach().cpu().float().contiguous()},
    }


def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    activation = dequantize_nvfp4(activation_quant, activation_scale).float()
    smooth = activation_state.get("smooth") if isinstance(activation_state, dict) else None
    if isinstance(smooth, torch.Tensor):
        activation = activation / smooth.to(device=activation.device, dtype=torch.float32)
    return _quantize_hif4_search(
        activation,
        importance=None,
        offsets=(-1, 0, 1, 2, 3, 4),
    )


def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    heads_per_kv = q_num_heads // kv_num_heads
    q_peak: torch.Tensor | None = None
    k_peak: torch.Tensor | None = None
    for sample in calib_qkv_list:
        q = dequantize_nvfp4(*sample["q"]).float()
        k = dequantize_nvfp4(*sample["k"]).float()
        current_q_peak = q.abs().reshape(-1, q_num_heads, head_dim).amax(dim=0)
        current_q_peak = current_q_peak.reshape(
            kv_num_heads, heads_per_kv, head_dim
        ).amax(dim=1)
        current_k_peak = k.abs().reshape(-1, kv_num_heads, head_dim).amax(dim=0)
        q_peak = current_q_peak if q_peak is None else torch.maximum(q_peak, current_q_peak)
        k_peak = current_k_peak if k_peak is None else torch.maximum(k_peak, current_k_peak)
    assert q_peak is not None and k_peak is not None

    # The same scale is shared by a KV head and all of its mapped Q heads:
    # (Q / smooth) @ (K * smooth)^T == Q @ K^T.
    smooth = torch.sqrt(q_peak.clamp_min(1.0e-8) / k_peak.clamp_min(1.0e-8))
    smooth.clamp_(1.0 / 16.0, 16.0)
    q_smooth = smooth.repeat_interleave(heads_per_kv, dim=0).reshape(-1)
    k_smooth = smooth.reshape(-1)
    return {
        "q_state": {"smooth": q_smooth.detach().cpu().float().contiguous()},
        "k_state": {"smooth": k_smooth.detach().cpu().float().contiguous()},
        "v_state": None,
    }


def _dynamic_attention_quantize(
    quant: torch.Tensor,
    scale: torch.Tensor,
    state: Any,
    inverse: bool = False,
) -> dict[str, torch.Tensor]:
    values = dequantize_nvfp4(quant, scale).float()
    smooth = state.get("smooth") if isinstance(state, dict) else None
    if isinstance(smooth, torch.Tensor):
        smooth = smooth.to(device=values.device, dtype=torch.float32)
        values = values * smooth if inverse else values / smooth
    return _quantize_hif4_search(
        values,
        importance=None,
        offsets=(-1, 0, 1, 2, 3, 4),
    )


def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    return _dynamic_attention_quantize(q_quant, q_scale, q_state)


def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    return _dynamic_attention_quantize(k_quant, k_scale, k_state, inverse=True)


def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    return _dynamic_attention_quantize(v_quant, v_scale, v_state)


# v2 output-aware calibration overrides.
_OFFSETS = (-1, 0, 1, 2, 3, 4)


def _quantize(x: torch.Tensor) -> dict[str, torch.Tensor]:
    return _quantize_hif4_search(x, importance=None, offsets=_OFFSETS)


def _dequantize(params: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        params["sign"]
        * params["mant"]
        * params["scale_lv2"]
        * params["scale_lv3"]
        * params["scale_factor"]
    ).reshape(*params["sign"].shape[:-4], -1)


def _linear_smooth(
    activation_peak: torch.Tensor,
    weight_peak: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    smooth = activation_peak.pow(alpha) / weight_peak.pow(1.0 - alpha)
    smooth = smooth / smooth.log().mean().exp()
    return smooth.clamp(1.0 / 16.0, 16.0)


def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    weight = dequantize_nvfp4(weight_quant, weight_scale).float()
    activations = [dequantize_nvfp4(q, s).float() for q, s in calib_activation_list]
    activation_peak = torch.stack(
        [value.abs().amax(dim=tuple(range(value.ndim - 1))) for value in activations]
    ).amax(dim=0).clamp_min_(1.0e-8)
    weight_peak = weight.abs().amax(dim=tuple(range(weight.ndim - 1))).clamp_min_(1.0e-8)

    # Output-row sampling keeps calibration-output reconstruction inexpensive.
    out_features = int(weight.shape[0])
    sample_count = min(out_features, 256)
    if sample_count == out_features:
        row_index = torch.arange(out_features, device=weight.device)
    else:
        row_index = torch.linspace(
            0, out_features - 1, sample_count, device=weight.device
        ).round().to(torch.int64)
    sampled_weight = weight.index_select(0, row_index)
    stacked_activation = torch.cat(activations, dim=0)
    reference = torch.matmul(stacked_activation, sampled_weight.transpose(-1, -2))

    # Only a scalar alpha is fitted, which limits calibration overfitting.
    alpha_candidates = (0.25, 0.5, 0.75, 0.875, 1.0, 1.125)
    best_alpha = 0.875
    best_error = float("inf")
    for alpha in alpha_candidates:
        smooth = _linear_smooth(activation_peak, weight_peak, alpha)
        quant_weight = _dequantize(_quantize(sampled_weight * smooth))
        quant_activation = _dequantize(_quantize(stacked_activation / smooth))
        reconstructed = torch.matmul(quant_activation, quant_weight.transpose(-1, -2))
        error = float((reconstructed - reference).square().mean().item())
        if error < best_error:
            best_error = error
            best_alpha = alpha

    smooth = _linear_smooth(activation_peak, weight_peak, best_alpha)
    weight_params = _quantize(weight * smooth)
    return {
        "weight_params": weight_params,
        "activation_state": {
            "smooth": smooth.detach().cpu().float().contiguous(),
            "alpha": float(best_alpha),
        },
    }


def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    activation = dequantize_nvfp4(activation_quant, activation_scale).float()
    smooth = activation_state["smooth"].to(device=activation.device, dtype=torch.float32)
    return _quantize(activation / smooth)


def _gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    seq_len = int(q.shape[0])
    qh = q.reshape(seq_len, q_num_heads, head_dim).transpose(0, 1)
    kh = k.reshape(seq_len, kv_num_heads, head_dim).transpose(0, 1)
    vh = v.reshape(seq_len, kv_num_heads, head_dim).transpose(0, 1)
    repeats = q_num_heads // kv_num_heads
    kh = kh.repeat_interleave(repeats, dim=0)
    vh = vh.repeat_interleave(repeats, dim=0)
    probabilities = torch.softmax(
        torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(head_dim), dim=-1
    )
    return torch.matmul(probabilities, vh)


def _attention_smooth(
    q_peak: torch.Tensor,
    k_peak: torch.Tensor,
    alpha: float,
    gain: float,
) -> torch.Tensor:
    smooth = q_peak.pow(alpha) / k_peak.pow(1.0 - alpha)
    return (smooth * gain).clamp(1.0 / 16.0, 16.0)


def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    heads_per_kv = q_num_heads // kv_num_heads
    decoded: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    q_peak: torch.Tensor | None = None
    k_peak: torch.Tensor | None = None
    for sample in calib_qkv_list:
        q = dequantize_nvfp4(*sample["q"]).float()
        k = dequantize_nvfp4(*sample["k"]).float()
        v = dequantize_nvfp4(*sample["v"]).float()
        decoded.append((q, k, v))
        current_q_peak = q.abs().reshape(-1, q_num_heads, head_dim).amax(dim=0)
        current_q_peak = current_q_peak.reshape(
            kv_num_heads, heads_per_kv, head_dim
        ).amax(dim=1)
        current_k_peak = k.abs().reshape(-1, kv_num_heads, head_dim).amax(dim=0)
        q_peak = current_q_peak if q_peak is None else torch.maximum(q_peak, current_q_peak)
        k_peak = current_k_peak if k_peak is None else torch.maximum(k_peak, current_k_peak)
    assert q_peak is not None and k_peak is not None
    q_peak.clamp_min_(1.0e-8)
    k_peak.clamp_min_(1.0e-8)

    references = [
        _gqa_attention(q, k, v, q_num_heads, kv_num_heads, head_dim)
        for q, k, v in decoded
    ]
    # V does not participate in the reversible Q/K transform, so quantize it once.
    quantized_v = [_dequantize(_quantize(v)) for _, _, v in decoded]
    best_alpha, best_gain, best_error = 0.5, 1.0, float("inf")
    for alpha in (0.25, 0.5, 0.75, 1.0):
        for gain in (0.875, 1.0, 1.125):
            smooth = _attention_smooth(q_peak, k_peak, alpha, gain)
            q_smooth = smooth.repeat_interleave(heads_per_kv, dim=0).reshape(-1)
            k_smooth = smooth.reshape(-1)
            error = 0.0
            for sample_index, (q, k, _) in enumerate(decoded):
                quantized_q = _dequantize(_quantize(q / q_smooth))
                quantized_k = _dequantize(_quantize(k * k_smooth))
                output = _gqa_attention(
                    quantized_q,
                    quantized_k,
                    quantized_v[sample_index],
                    q_num_heads,
                    kv_num_heads,
                    head_dim,
                )
                error += float((output - references[sample_index]).square().mean().item())
            if error < best_error:
                best_error = error
                best_alpha, best_gain = alpha, gain

    smooth = _attention_smooth(q_peak, k_peak, best_alpha, best_gain)
    q_smooth = smooth.repeat_interleave(heads_per_kv, dim=0).reshape(-1)
    k_smooth = smooth.reshape(-1)
    return {
        "q_state": {
            "smooth": q_smooth.detach().cpu().float().contiguous(),
            "alpha": float(best_alpha),
            "gain": float(best_gain),
        },
        "k_state": {"smooth": k_smooth.detach().cpu().float().contiguous()},
        "v_state": None,
    }


def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    q = dequantize_nvfp4(q_quant, q_scale).float()
    smooth = q_state["smooth"].to(device=q.device, dtype=torch.float32)
    return _quantize(q / smooth)


def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    k = dequantize_nvfp4(k_quant, k_scale).float()
    smooth = k_state["smooth"].to(device=k.device, dtype=torch.float32)
    return _quantize(k * smooth)


def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    return _quantize(dequantize_nvfp4(v_quant, v_scale).float())
