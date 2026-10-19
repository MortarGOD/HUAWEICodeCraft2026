"""Candidate v1: task-aware hierarchical HiF4 quantization.

The public HiF4 representation is fixed, but its hierarchical scales are not
uniquely determined.  This implementation searches a small E6M2 neighbourhood
and exactly minimizes a diagonal quadratic objective over the E1_8/E1_16 tree.
It uses only PyTorch and keeps all calibration state as finite CPU tensors.
"""

from __future__ import annotations

from typing import Any, Iterable

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


# RRS-inspired v4 overrides. Later public definitions intentionally replace v1.
_OFFSETS = (-1, 0, 1, 2, 3, 4)


def _quantize(x: torch.Tensor) -> dict[str, torch.Tensor]:
    return _quantize_hif4_search(x, None, _OFFSETS)


def _proxy_quantize(x: torch.Tensor) -> dict[str, torch.Tensor]:
    return _quantize_hif4_search(x, None, (0,))


def _dequantize(p: dict[str, torch.Tensor]) -> torch.Tensor:
    x = p["sign"] * p["mant"] * p["scale_lv2"] * p["scale_lv3"] * p["scale_factor"]
    return x.reshape(*p["sign"].shape[:-4], -1)


def _fwht(x: torch.Tensor, block_size: int) -> torch.Tensor:
    channels = int(x.shape[-1])
    if block_size < 1 or block_size & (block_size - 1) or channels % block_size:
        raise ValueError("invalid FWHT block size")
    prefix = tuple(int(v) for v in x.shape[:-1])
    blocks = x.reshape(*prefix, channels // block_size, block_size)
    width = 1
    while width < block_size:
        paired = blocks.reshape(
            *prefix, channels // block_size, block_size // (2 * width), 2, width
        )
        left, right = paired[..., 0, :], paired[..., 1, :]
        blocks = torch.cat((left + right, left - right), dim=-1).reshape(
            *prefix, channels // block_size, block_size
        )
        width *= 2
    return blocks.reshape(*prefix, channels) * (block_size ** -0.5)


def _make_sign(channels: int, seed: int, device: torch.device) -> torch.Tensor:
    if seed == 0:
        return torch.ones(channels, dtype=torch.float32, device=device)
    index = torch.arange(channels, dtype=torch.int64, device=device)
    bit = torch.bitwise_and((index * 1103515245 + seed * 12345) >> 16, 1)
    return bit.float().mul_(2.0).sub_(1.0).contiguous()


def _block_sizes(channels: int) -> tuple[int, ...]:
    largest = 1
    while largest * 2 <= min(channels, 2048) and channels % (largest * 2) == 0:
        largest *= 2
    values = {64, largest}
    if channels % 256 == 0:
        values.add(256)
    return tuple(sorted(v for v in values if channels % v == 0))


def _transform(
    x: torch.Tensor, permutation: torch.Tensor, sign: torch.Tensor, block_size: int
) -> torch.Tensor:
    permutation = permutation.to(x.device, torch.int64)
    sign = sign.to(x.device, torch.float32)
    return _fwht(x.index_select(-1, permutation) * sign, block_size)


def _split(samples: list) -> tuple[list, list]:
    return (samples, samples) if len(samples) <= 1 else (samples[:-1], samples[-1:])


def _attention_split(samples: list) -> tuple[list, list]:
    fit, valid = _split(samples)
    # Keep v7's fit statistics unchanged, but require each transform to
    # reconstruct two calibration outputs instead of over-specializing to one.
    if len(samples) >= 3:
        valid = samples[-2:]
    return fit, valid


def _smooth(a_peak: torch.Tensor, w_peak: torch.Tensor) -> torch.Tensor:
    alpha = 0.875
    scale = a_peak.clamp_min(1e-8).pow(alpha) / w_peak.clamp_min(1e-8).pow(1 - alpha)
    scale = scale / scale.log().mean().exp()
    return scale.clamp(1 / 16, 16)


def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor, weight_scale: torch.Tensor, calib_activation_list: list
) -> dict[str, Any]:
    weight = dequantize_nvfp4(weight_quant, weight_scale).float()
    fit_pairs, valid_pairs = _split(calib_activation_list)
    fit = torch.cat([dequantize_nvfp4(q, s).float() for q, s in fit_pairs])
    valid = torch.cat([dequantize_nvfp4(q, s).float() for q, s in valid_pairs])
    channels = int(weight.shape[-1])
    identity = torch.arange(channels, device=weight.device)
    sorted_perm = torch.argsort(fit.abs().amax(0))
    rows = min(int(weight.shape[0]), 128)
    row_index = (
        torch.arange(rows, device=weight.device)
        if rows == weight.shape[0]
        else torch.linspace(0, weight.shape[0] - 1, rows, device=weight.device).round().long()
    )
    sampled_weight = weight.index_select(0, row_index)
    reference = valid @ sampled_weight.T
    best_error = float("inf")
    best = (identity, _make_sign(channels, 0, weight.device), 64)
    for permutation in (identity, sorted_perm):
        for seed in (0, 1):
            sign = _make_sign(channels, seed, weight.device)
            for block_size in _block_sizes(channels):
                qw = _dequantize(_proxy_quantize(_transform(sampled_weight, permutation, sign, block_size)))
                qa = _dequantize(_proxy_quantize(_transform(valid, permutation, sign, block_size)))
                error = float(((qa @ qw.T) - reference).square().mean())
                if error < best_error:
                    best_error, best = error, (permutation, sign, block_size)
    permutation, sign, block_size = best
    transformed_weight = _transform(weight, permutation, sign, block_size)
    transformed_fit = _transform(fit, permutation, sign, block_size)
    smooth = _smooth(transformed_fit.abs().amax(0), transformed_weight.abs().amax(0))
    return {
        "weight_params": _quantize(transformed_weight * smooth),
        "activation_state": {
            "permutation": permutation.cpu().to(torch.int32).contiguous(),
            "sign": sign.cpu().to(torch.int8).contiguous(),
            "block_size": int(block_size),
            "smooth": smooth.cpu().float().contiguous(),
        },
    }


def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor, activation_scale: torch.Tensor, activation_state: Any
) -> dict[str, torch.Tensor]:
    x = dequantize_nvfp4(activation_quant, activation_scale).float()
    x = _transform(x, activation_state["permutation"], activation_state["sign"], activation_state["block_size"])
    return _quantize(x / activation_state["smooth"].to(x.device))


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qh: int, kh: int, d: int) -> torch.Tensor:
    n = int(q.shape[0])
    q = q.reshape(n, qh, d).transpose(0, 1)
    k = k.reshape(n, kh, d).transpose(0, 1).repeat_interleave(qh // kh, 0)
    v = v.reshape(n, kh, d).transpose(0, 1).repeat_interleave(qh // kh, 0)
    return torch.softmax((q @ k.transpose(-1, -2)) / (d ** 0.5), -1) @ v


def _head_transform(x: torch.Tensor, heads: int, d: int, p: torch.Tensor, s: torch.Tensor, b: int) -> torch.Tensor:
    n = int(x.shape[0])
    return _transform(x.reshape(n, heads, d), p, s, b).reshape(n, heads * d)


def hif4_calibration_attention(calib_qkv_list: list, q_num_heads: int, kv_num_heads: int, head_dim: int) -> dict[str, Any]:
    fit_samples, valid_samples = _attention_split(calib_qkv_list)
    ratio = q_num_heads // kv_num_heads
    fit_q = torch.cat([dequantize_nvfp4(*z["q"]).float() for z in fit_samples])
    fit_k = torch.cat([dequantize_nvfp4(*z["k"]).float() for z in fit_samples])
    q_peak = fit_q.abs().reshape(-1, q_num_heads, head_dim).amax((0, 1))
    k_peak = fit_k.abs().reshape(-1, kv_num_heads, head_dim).amax((0, 1))
    identity = torch.arange(head_dim, device=fit_q.device)
    sorted_perm = torch.argsort(torch.maximum(q_peak, k_peak))
    decoded = [(dequantize_nvfp4(*z["q"]).float(), dequantize_nvfp4(*z["k"]).float(), dequantize_nvfp4(*z["v"]).float()) for z in valid_samples]
    references = [_attention(q, k, v, q_num_heads, kv_num_heads, head_dim) for q, k, v in decoded]
    qv = [_dequantize(_proxy_quantize(v)) for _, _, v in decoded]
    best_error = float("inf")
    best = (identity, _make_sign(head_dim, 0, fit_q.device), 64)
    best_smooth: torch.Tensor | None = None
    for permutation in (identity, sorted_perm):
        for seed in (0, 1):
            sign = _make_sign(head_dim, seed, fit_q.device)
            for block_size in _block_sizes(head_dim):
                # Rank each transform through the same reciprocal smoothing used
                # by the deployed Q/K quantizers.  v6 ranked unsmoothed proxies
                # and only introduced this scale after choosing a transform.
                fit_tq = _head_transform(
                    fit_q, q_num_heads, head_dim, permutation, sign, block_size
                )
                fit_tk = _head_transform(
                    fit_k, kv_num_heads, head_dim, permutation, sign, block_size
                )
                candidate_q_peak = (
                    fit_tq.abs()
                    .reshape(-1, q_num_heads, head_dim)
                    .amax(0)
                    .reshape(kv_num_heads, ratio, head_dim)
                    .amax(1)
                )
                candidate_k_peak = (
                    fit_tk.abs().reshape(-1, kv_num_heads, head_dim).amax(0)
                )
                candidate_smooth = torch.sqrt(
                    candidate_q_peak.clamp_min(1e-8)
                    / candidate_k_peak.clamp_min(1e-8)
                ).clamp(1 / 16, 16)
                q_smooth = candidate_smooth.repeat_interleave(ratio, 0).reshape(-1)
                k_smooth = candidate_smooth.reshape(-1)
                error = 0.0
                for i, (q, k, _) in enumerate(decoded):
                    tq = _head_transform(q, q_num_heads, head_dim, permutation, sign, block_size)
                    tk = _head_transform(k, kv_num_heads, head_dim, permutation, sign, block_size)
                    out = _attention(
                        _dequantize(_proxy_quantize(tq / q_smooth)),
                        _dequantize(_proxy_quantize(tk * k_smooth)),
                        qv[i],
                        q_num_heads,
                        kv_num_heads,
                        head_dim,
                    )
                    error += float((out - references[i]).square().mean())
                if error < best_error:
                    best_error, best = error, (permutation, sign, block_size)
                    best_smooth = candidate_smooth
    permutation, sign, block_size = best
    assert best_smooth is not None
    smooth = best_smooth
    common = {"permutation": permutation.cpu().to(torch.int32).contiguous(), "sign": sign.cpu().to(torch.int8).contiguous(), "block_size": int(block_size)}
    return {
        "q_state": {**common, "smooth": smooth.repeat_interleave(ratio, 0).reshape(-1).cpu().float().contiguous()},
        "k_state": {**common, "smooth": smooth.reshape(-1).cpu().float().contiguous()},
        "v_state": None,
    }


def hif4_dynamic_quantize_q(q_quant: torch.Tensor, q_scale: torch.Tensor, q_num_heads: int, head_dim: int, q_state: Any) -> dict[str, torch.Tensor]:
    q = dequantize_nvfp4(q_quant, q_scale).float()
    q = _head_transform(q, q_num_heads, head_dim, q_state["permutation"], q_state["sign"], q_state["block_size"])
    return _quantize(q / q_state["smooth"].to(q.device))


def hif4_dynamic_quantize_k(k_quant: torch.Tensor, k_scale: torch.Tensor, kv_num_heads: int, head_dim: int, k_state: Any) -> dict[str, torch.Tensor]:
    k = dequantize_nvfp4(k_quant, k_scale).float()
    k = _head_transform(k, kv_num_heads, head_dim, k_state["permutation"], k_state["sign"], k_state["block_size"])
    return _quantize(k * k_state["smooth"].to(k.device))


def hif4_dynamic_quantize_v(v_quant: torch.Tensor, v_scale: torch.Tensor, kv_num_heads: int, head_dim: int, v_state: Any) -> dict[str, torch.Tensor]:
    return _quantize(dequantize_nvfp4(v_quant, v_scale).float())


def _linear_smooth_alpha(
    activation_peak: torch.Tensor,
    weight_peak: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    smooth = activation_peak.clamp_min(1e-8).pow(alpha)
    smooth = smooth / weight_peak.clamp_min(1e-8).pow(1.0 - alpha)
    smooth = smooth / smooth.log().mean().exp()
    return smooth.clamp(1.0 / 16.0, 16.0)


# v11 Linear: select SmoothQuant alpha with calibration-only A@W reconstruction.
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    weight = dequantize_nvfp4(weight_quant, weight_scale).float()
    fit_pairs, validation_pairs = _split(calib_activation_list)
    fit_activations = [
        dequantize_nvfp4(quant, scale).float() for quant, scale in fit_pairs
    ]
    validation = torch.cat(
        [
            dequantize_nvfp4(quant, scale).float()
            for quant, scale in validation_pairs
        ],
        dim=0,
    )
    activation_peak = torch.stack(
        [
            value.abs().amax(dim=tuple(range(value.ndim - 1)))
            for value in fit_activations
        ]
    ).amax(dim=0)
    weight_peak = weight.abs().amax(dim=tuple(range(weight.ndim - 1))).clamp_min_(1.0e-8)
    activation_peak.clamp_min_(1.0e-8)

    out_features = int(weight.shape[0])
    sample_count = min(out_features, 128)
    if sample_count == out_features:
        row_index = torch.arange(out_features, device=weight.device)
    else:
        row_index = torch.linspace(
            0, out_features - 1, sample_count, device=weight.device
        ).round().to(torch.int64)
    sampled_weight = weight.index_select(0, row_index)
    reference = validation @ sampled_weight.transpose(-1, -2)

    best_alpha = 0.875
    best_error = float("inf")
    for alpha in (0.25, 0.5, 0.75, 0.875, 1.0, 1.125):
        smooth = _linear_smooth_alpha(activation_peak, weight_peak, alpha)
        quant_weight = _dequantize(_proxy_quantize(sampled_weight * smooth))
        quant_activation = _dequantize(_proxy_quantize(validation / smooth))
        reconstructed = quant_activation @ quant_weight.transpose(-1, -2)
        error = float((reconstructed - reference).square().mean())
        if error < best_error:
            best_error = error
            best_alpha = alpha

    smooth = _linear_smooth_alpha(activation_peak, weight_peak, best_alpha)
    offset_candidates = (
        (-3, -2, -1, 0),
        (-2, -1, 0, 1, 2),
        _OFFSETS,
    )
    best_offsets = _OFFSETS
    best_offset_error = float("inf")
    for offsets in offset_candidates:
        quant_weight = _dequantize(
            _quantize_hif4_search(sampled_weight * smooth, None, offsets)
        )
        quant_activation = _dequantize(
            _quantize_hif4_search(validation / smooth, None, offsets)
        )
        reconstructed = quant_activation @ quant_weight.transpose(-1, -2)
        error = float((reconstructed - reference).square().mean())
        if error < best_offset_error:
            best_offset_error = error
            best_offsets = offsets

    weight_params = _quantize_hif4_search(weight * smooth, None, best_offsets)
    return {
        "weight_params": weight_params,
        "activation_state": {
            "smooth": smooth.detach().cpu().float().contiguous(),
            "alpha": float(best_alpha),
            "offsets": tuple(int(offset) for offset in best_offsets),
        },
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
    offsets = (
        activation_state.get("offsets", _OFFSETS)
        if isinstance(activation_state, dict)
        else _OFFSETS
    )
    return _quantize_hif4_search(
        activation,
        importance=None,
        offsets=offsets,
    )
