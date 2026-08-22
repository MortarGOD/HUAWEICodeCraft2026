"""Version 29: cxw_v1 Linear mechanisms with the proven v28 Attention path."""

from __future__ import annotations

from typing import Any, Iterable

import torch


_SCALE_OFFSETS = (0, 1, 2, 3, 4)
_QK_SCALE_OFFSETS = (0, 1, 3, 4)
_V_SCALE_OFFSETS = (0, 1, 2, 3)
_LINEAR_HADAMARD = 64
_QK_NEAR_RMS_RATIO = 0.01
_QK_SPARSE_THRESHOLD = 0.25
_LINEAR_BALANCE_EXPONENT = 0.39
_LINEAR_RIDGE_LAMBDAS = (10.0, 20.0)
_LINEAR_RIDGE_SWITCH_MARGIN = 0.05
_WEIGHT_REFINEMENT_RIDGE = 0.01
_WEIGHT_REFINEMENT_STEP = 0.04
_WEIGHT_REFINEMENT_CHECKPOINTS = (1, 2, 3, 4, 5)
_WEIGHT_REFINEMENT_FINAL_REQUANTIZE = True
_Q_MOMENT_PENALTIES = (0.0, 0.25)
_K_MOMENT_PENALTIES = (0.0, 0.25)
_V_MOMENT_PENALTIES = (0.0, 0.0)
_V_CONFIGS = (
    ((0, 1, 2, 3), 0.0, 0.0),
    ((-1, 0, 1, 2, 3), 0.0, 0.0),
    ((0, 1, 2, 3), 0.125, 0.0),
    ((0, 1, 2, 3), 0.0, 0.125),
    ((-1, 0, 1, 2, 3), 0.0, 0.125),
)
_GPTQ_RIDGE = 0.1
_GPTQ_USE_ALL_CALIBRATION = True
_GPTQ_SELECT_ROWS = False
_GPTQ_WEIGHT_IMPORTANCE = False
_GPTQ_FULL_PROPAGATION_MAX_GROUPS = 16
_GPTQ_WIDE_PROPAGATION_GROUPS = 8
_GPTQ_FULL_PROPAGATION_MAX_GROUP_ROWS = 8192
_GPTQ_CONTIGUOUS_ORDERED_WORK = True
_GPTQ_TALL_OUTPUT_ROWS = 1024
_GPTQ_TALL_PROPAGATION_GROUPS = 4
_LINEAR_SIGN_WEIGHT_ROWS = 512
_GPTQ_LARGE_WEIGHT_OFFSETS = (0, 1, 2, 3)
_GPTQ_SKIP_UNUSED_BASELINE = True

_ATTENTION_QK_OFFSETS = (0, 1, 2, 3, 4)
_ATTENTION_V_OFFSETS = (-1, 0, 1, 2, 3)
_ATTENTION_ENERGY_PENALTY = 0.125
_ATTENTION_NEAR_ZERO_RATIO = 0.01
_ATTENTION_SPARSE_THRESHOLD = 0.25


def dequantize_nvfp4(
    quant_float: torch.Tensor, scale_float: torch.Tensor, blk_size: int = 16
) -> torch.Tensor:
    values = quant_float.unflatten(-1, (-1, blk_size))
    return (values * scale_float.unsqueeze(-1)).flatten(-2, -1).to(torch.bfloat16)


def _decode_e6m2(code: torch.Tensor) -> torch.Tensor:
    code64 = code.to(torch.int64)
    mantissa = 1.0 + (code64 & 3).to(torch.float32) / 4.0
    return torch.ldexp(mantissa, (code64 >> 2) - 48)


def _encode_e6m2(scale: torch.Tensor) -> torch.Tensor:
    positive = scale > 0
    safe = torch.where(positive, scale, torch.ones_like(scale))
    exponent = torch.floor(torch.log2(safe)).to(torch.int64)
    normalized = torch.ldexp(safe, -exponent)
    mantissa = torch.round((normalized - 1.0) * 4.0).to(torch.int64)
    carry = mantissa == 4
    exponent = (exponent + carry.to(torch.int64)).clamp(-48, 15)
    mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa).clamp(0, 3)
    code = (((exponent + 48) << 2) | mantissa).clamp(0, 0xFE)
    return torch.where(positive, code, torch.full_like(code, 48 << 2)).to(torch.uint8)


def _best_for_base(
    groups: torch.Tensor,
    scale_code: torch.Tensor,
    weights: torch.Tensor | None = None,
    mean_penalty: float = 0.0,
    norm_penalty: float = 0.0,
) -> tuple[torch.Tensor, ...]:
    base = _decode_e6m2(scale_code)
    factors = groups.new_tensor((1.0, 2.0, 4.0)).view(
        *([1] * (groups.ndim - 2)), 3, 1, 1
    )
    scaled = groups.unsqueeze(-3) / base.unsqueeze(-3) / factors
    negative = scaled < 0
    mantissa = torch.round(scaled.abs().clamp(max=1.75) * 4.0).to(torch.uint8).clamp(max=7)
    reconstructed = (
        base.unsqueeze(-3)
        * factors
        * torch.where(negative, -1.0, 1.0)
        * mantissa.to(torch.float32)
        / 4.0
    )
    target = groups.unsqueeze(-3)
    residual = target - reconstructed
    errors = residual.square()
    if weights is not None:
        errors = errors * weights.unsqueeze(-3)
    errors = errors.sum(dim=-1)
    error0, lv30 = errors[..., :2, :].min(dim=-2)
    error1, lv31 = errors[..., 1:, :].min(dim=-2)
    use1 = error1.sum(dim=-1) < error0.sum(dim=-1)
    lv2 = use1.to(torch.uint8)
    lv3 = torch.where(use1.unsqueeze(-1), lv31, lv30).to(torch.uint8)
    factor_index = (lv2.unsqueeze(-1) + lv3).to(torch.int64)
    index = factor_index.unsqueeze(-2).unsqueeze(-1).expand(
        *factor_index.shape[:-1], 1, 2, 4
    )
    selected_reconstruction = torch.gather(reconstructed, -3, index).squeeze(-3)
    selected_residual = groups - selected_reconstruction
    selected_errors = selected_residual.square()
    if weights is not None:
        selected_errors = selected_errors * weights
    score = selected_errors.sum(dim=(-3, -2, -1))
    if mean_penalty:
        score = score + mean_penalty * selected_residual.sum(
            dim=(-3, -2, -1)
        ).square()
    if norm_penalty:
        target_energy = groups.square().sum(dim=(-3, -2, -1))
        reconstructed_energy = selected_reconstruction.square().sum(
            dim=(-3, -2, -1)
        )
        score = score + norm_penalty * (
            target_energy - reconstructed_energy
        ).square() / target_energy.clamp_min(torch.finfo(torch.float32).tiny)
    return (
        score,
        scale_code,
        lv2.unsqueeze(-1).unsqueeze(-1),
        lv3.unsqueeze(-1),
        torch.gather(negative, -3, index).squeeze(-3),
        torch.gather(mantissa, -3, index).squeeze(-3),
    )


def _quantize_values(
    values: torch.Tensor,
    importance: torch.Tensor | None = None,
    offsets: Iterable[int] = _SCALE_OFFSETS,
    mean_penalty: float = 0.0,
    norm_penalty: float = 0.0,
) -> dict[str, torch.Tensor]:
    channels = int(values.shape[-1])
    groups = values.to(torch.float32).reshape(
        *values.shape[:-1], channels // 64, 8, 2, 4
    )
    weights = None
    if importance is not None:
        weights = importance.to(torch.float32).reshape(1, channels // 64, 8, 2, 4)
    peak_code = _encode_e6m2(groups.abs().amax(dim=(-3, -2, -1), keepdim=True) / 7.0)
    best = None
    for offset in offsets:
        code = (peak_code.to(torch.int16) + int(offset)).clamp(0, 0xFE).to(torch.uint8)
        current = _best_for_base(
            groups, code, weights, mean_penalty, norm_penalty
        )
        if best is None:
            best = current
            continue
        choose = current[0] < best[0]
        mask = choose[..., None, None, None]
        best = (torch.where(choose, current[0], best[0]),) + tuple(
            torch.where(mask, new, old) for new, old in zip(current[1:], best[1:])
        )
    _, code, lv2, lv3, negative, mantissa_code = best
    mantissa = mantissa_code.to(torch.bfloat16) / 4.0
    sign = torch.where(
        mantissa == 0,
        torch.zeros_like(mantissa),
        torch.where(negative, -torch.ones_like(mantissa), torch.ones_like(mantissa)),
    )
    return {
        "scale_factor": _decode_e6m2(code).to(torch.bfloat16),
        "scale_lv2": 1.0 + lv2.to(torch.bfloat16),
        "scale_lv3": 1.0 + lv3.to(torch.bfloat16),
        "sign": sign,
        "mant": mantissa,
    }


def _dequantize_hif4(params: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        params["scale_factor"].to(torch.float32)
        * params["scale_lv2"].to(torch.float32)
        * params["scale_lv3"].to(torch.float32)
        * params["sign"].to(torch.float32)
        * params["mant"].to(torch.float32)
    ).flatten(-4, -1)


def _gather_offset(t: torch.Tensor, best_off: torch.Tensor) -> torch.Tensor:
    rest = t.shape[1 + best_off.ndim:]
    idx = best_off.reshape((1,) + best_off.shape + (1,) * len(rest)).expand(
        (1,) + best_off.shape + rest
    )
    return torch.gather(t, 0, idx)[0]


def _quantize_values_vec(
    values: torch.Tensor,
    importance: torch.Tensor | None = None,
    offsets: Iterable[int] = _SCALE_OFFSETS,
    mean_penalty: float = 0.0,
    norm_penalty: float = 0.0,
) -> dict[str, torch.Tensor]:
    channels = int(values.shape[-1])
    groups = values.to(torch.float32).reshape(
        *values.shape[:-1], channels // 64, 8, 2, 4
    )
    lead = groups.shape[:-4]
    weights = None
    if importance is not None:
        weights = importance.to(torch.float32).reshape(1, channels // 64, 8, 2, 4)
    peak = groups.abs().amax(dim=(-3, -2, -1), keepdim=True) / 7.0
    peak_code = _encode_e6m2(peak)
    offs = torch.tensor(list(offsets), device=groups.device)
    codes = (
        peak_code.unsqueeze(0).to(torch.int16)
        + offs.view(-1, *([1] * groups.ndim))
    ).clamp(0, 0xFE).to(torch.uint8)
    base = _decode_e6m2(codes)
    g = groups.unsqueeze(0).unsqueeze(-3)
    b = base.unsqueeze(-3)
    f = groups.new_tensor((1.0, 2.0, 4.0)).view(*([1] * (g.ndim - 3)), 3, 1, 1)
    scaled = g / b / f
    negative = scaled < 0
    mantissa = torch.round(scaled.abs().clamp(max=1.75) * 4.0).to(torch.uint8).clamp(max=7)
    reconstructed = (
        b * f * torch.where(negative, -1.0, 1.0) * mantissa.to(torch.float32) / 4.0
    )
    residual = g - reconstructed
    errors = residual.square()
    if weights is not None:
        errors = errors * weights.view(1, *([1] * len(lead)), channels // 64, 8, 1, 2, 4)
    errors = errors.sum(dim=-1)
    error0, lv30 = errors[..., :2, :].min(dim=-2)
    error1, lv31 = errors[..., 1:, :].min(dim=-2)
    use1 = error1.sum(dim=-1) < error0.sum(dim=-1)
    lv2 = use1.to(torch.uint8)
    lv3 = torch.where(use1.unsqueeze(-1), lv31, lv30).to(torch.uint8)
    factor_index = (lv2.unsqueeze(-1) + lv3).to(torch.int64)
    index = factor_index.unsqueeze(-2).unsqueeze(-1).expand(
        *factor_index.shape[:-1], 1, 2, 4
    )
    selected_reconstruction = torch.gather(reconstructed, -3, index).squeeze(-3)
    selected_residual = groups - selected_reconstruction
    selected_errors = selected_residual.square()
    if weights is not None:
        selected_errors = selected_errors * weights.view(
            1, *([1] * len(lead)), channels // 64, 8, 2, 4
        )
    score = selected_errors.sum(dim=(-3, -2, -1))
    if mean_penalty:
        score = score + mean_penalty * selected_residual.sum(
            dim=(-3, -2, -1)
        ).square()
    if norm_penalty:
        target_energy = groups.square().sum(dim=(-3, -2, -1))
        reconstructed_energy = selected_reconstruction.square().sum(
            dim=(-3, -2, -1)
        )
        score = score + norm_penalty * (
            target_energy - reconstructed_energy
        ).square() / target_energy.clamp_min(torch.finfo(torch.float32).tiny)
    best_off = score.argmin(0)
    neg_all = torch.gather(negative, -3, index).squeeze(-3)
    mant_all = torch.gather(mantissa, -3, index).squeeze(-3)
    code_g = _gather_offset(codes, best_off)
    lv2_g = _gather_offset(lv2.unsqueeze(-1).unsqueeze(-1), best_off)
    lv3_g = _gather_offset(lv3.unsqueeze(-1), best_off)
    neg_g = _gather_offset(neg_all, best_off)
    mant_g = _gather_offset(mant_all, best_off)
    mantissa_f = mant_g.to(torch.bfloat16) / 4.0
    sign = torch.where(
        mantissa_f == 0,
        torch.zeros_like(mantissa_f),
        torch.where(neg_g, -torch.ones_like(mantissa_f), torch.ones_like(mantissa_f)),
    )
    return {
        "scale_factor": _decode_e6m2(code_g).to(torch.bfloat16),
        "scale_lv2": 1.0 + lv2_g.to(torch.bfloat16),
        "scale_lv3": 1.0 + lv3_g.to(torch.bfloat16),
        "sign": sign,
        "mant": mantissa_f,
    }


def _choose_weight_rows(
    target_output: torch.Tensor,
    quantized_activation: torch.Tensor,
    candidates: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    weights = torch.stack([_dequantize_hif4(params) for params in candidates])
    outputs = torch.matmul(quantized_activation, weights.transpose(1, 2))
    choices = (target_output.unsqueeze(0) - outputs).square().mean(1).argmin(0)
    rows = torch.arange(choices.numel())
    return {
        name: torch.stack([params[name] for params in candidates])[choices, rows]
        for name in candidates[0]
    }


def _block_gptq_weight(
    quantized_activation: torch.Tensor, weight: torch.Tensor
) -> dict[str, torch.Tensor]:
    channels = weight.shape[-1]
    group_count = channels // 64
    pieces: dict[str, list[torch.Tensor | None]] = {}
    tiny = torch.finfo(torch.float32).tiny
    order = _gptq_group_order(quantized_activation)
    ordered_indices = torch.cat(
        [torch.arange(group * 64, (group + 1) * 64) for group in order]
    )
    ordered_activation = quantized_activation.index_select(-1, ordered_indices)
    work = weight.to(torch.float32).index_select(-1, ordered_indices).clone()
    propagation_groups = _gptq_propagation_groups(group_count, work.shape[0])
    weight_offsets = _gptq_weight_offsets(group_count, work.shape[0])
    for position, group in enumerate(order):
        start = position * 64
        stop = start + 64
        activation_block = ordered_activation[..., start:stop]
        params = _quantize_values(work[..., start:stop], offsets=weight_offsets)
        quantized_block = _dequantize_hif4(params)
        for name, tensor in params.items():
            pieces.setdefault(name, [None] * group_count)[group] = tensor
        future_stop = min(channels, stop + propagation_groups * 64)
        if future_stop == stop:
            continue
        future_activation = ordered_activation[..., stop:future_stop]
        residual = activation_block @ (work[..., start:stop] - quantized_block).T
        kernel = future_activation @ future_activation.T
        ridge = _GPTQ_RIDGE * kernel.diag().mean().clamp_min(tiny)
        dual = torch.linalg.solve(
            kernel + ridge * torch.eye(kernel.shape[0]), residual
        )
        work[..., stop:future_stop] += (future_activation.T @ dual).T
    return {
        name: torch.cat([tensor for tensor in tensors if tensor is not None], dim=-4)
        for name, tensors in pieces.items()
    }


def _gptq_group_order(activation: torch.Tensor) -> list[int]:
    groups = activation.shape[-1] // 64
    energy = activation.to(torch.float32).reshape(-1, groups, 64).square().sum((0, 2))
    return energy.argsort(descending=True).tolist()


def _gptq_propagation_groups(group_count: int, output_rows: int) -> int:
    if (
        group_count <= _GPTQ_FULL_PROPAGATION_MAX_GROUPS
        and group_count * output_rows <= _GPTQ_FULL_PROPAGATION_MAX_GROUP_ROWS
    ):
        return group_count
    if output_rows >= _GPTQ_TALL_OUTPUT_ROWS:
        return _GPTQ_TALL_PROPAGATION_GROUPS
    return _GPTQ_WIDE_PROPAGATION_GROUPS


def _gptq_weight_offsets(group_count: int, output_rows: int) -> tuple[int, ...]:
    if (
        group_count <= _GPTQ_FULL_PROPAGATION_MAX_GROUPS
        and group_count * output_rows <= _GPTQ_FULL_PROPAGATION_MAX_GROUP_ROWS
    ):
        return _SCALE_OFFSETS
    return _GPTQ_LARGE_WEIGHT_OFFSETS


def _output_aware_quantize_weight(
    activation: torch.Tensor,
    weight: torch.Tensor,
    baseline: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    # Q(A) is finalized before either output is formed. Only returned W fields vary.
    activation_params = _quantize_values(activation)
    quantized_activation = _dequantize_hif4(activation_params)
    activation_importance = activation.square().mean(0)
    candidates = [
        baseline,
        _quantize_values(weight),
    ]
    target_output = activation @ weight.T
    initial_weights = torch.stack([_dequantize_hif4(params) for params in candidates])
    initial_outputs = torch.matmul(
        quantized_activation, initial_weights.transpose(1, 2)
    )
    initial_errors = (target_output.unsqueeze(0) - initial_outputs).square().mean(1)
    rows = torch.arange(weight.shape[0])
    best_output = initial_outputs[initial_errors.argmin(0), :, rows].T
    for source_output in (initial_outputs[0], best_output):
        denominator = source_output.square().sum(0).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        gain = (target_output * source_output).sum(0) / denominator
        candidates.append(
            _quantize_values(
                weight * gain.clamp(0.875, 1.125).unsqueeze(1),
                activation_importance,
            )
        )
    kernel = quantized_activation @ quantized_activation.T
    residual = target_output - quantized_activation @ weight.T
    ridge_scale = kernel.diag().mean()
    identity = torch.eye(
        kernel.shape[0], dtype=kernel.dtype, device=kernel.device
    )
    ridge_candidates = []
    for ridge_lambda in _LINEAR_RIDGE_LAMBDAS:
        dual = torch.linalg.solve(
            kernel + ridge_lambda * ridge_scale * identity, residual
        )
        fitted_weight = weight + (quantized_activation.T @ dual).T
        ridge_candidates.append(
            _quantize_values(fitted_weight, activation_importance)
        )
    row_params = [
        _choose_weight_rows(
            target_output, quantized_activation, candidates + [ridge_candidate]
        )
        for ridge_candidate in ridge_candidates
    ]
    row_errors = [
        (
            target_output
            - quantized_activation @ _dequantize_hif4(params).T
        ).square().mean(0)
        for params in row_params
    ]
    use_lower_ridge = row_errors[0] <= row_errors[1] * (
        1.0 - _LINEAR_RIDGE_SWITCH_MARGIN
    )
    output_count = weight.shape[0]
    return {
        name: torch.where(
            use_lower_ridge.view(output_count, *([1] * (lower.ndim - 1))),
            lower,
            row_params[1][name],
        )
        for name, lower in row_params[0].items()
    }


def _refine_weight_codes(
    activation_parts: list[torch.Tensor],
    transformed_activation_parts: list[torch.Tensor],
    weight: torch.Tensor,
    weight_params: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if len(activation_parts) < 2:
        return weight_params

    quantized_parts = [
        _dequantize_hif4(_quantize_values(part))
        for part in transformed_activation_parts
    ]
    train_activation = torch.cat(activation_parts[:-1])
    train_quantized = torch.cat(quantized_parts[:-1])
    train_target = train_activation @ weight.T
    energy = train_quantized.square().sum(0)
    denominator = energy + _WEIGHT_REFINEMENT_RIDGE * energy.mean()

    scalar_scale = (
        weight_params["scale_factor"].to(torch.float32)
        * weight_params["scale_lv2"].to(torch.float32)
        * weight_params["scale_lv3"].to(torch.float32)
    ).expand_as(weight_params["mant"]).flatten(-4, -1)
    candidate = dict(weight_params)
    checkpoints = [weight_params]
    for iteration in range(1, _WEIGHT_REFINEMENT_CHECKPOINTS[-1] + 1):
        candidate_weight = _dequantize_hif4(candidate)
        residual = train_target - train_quantized @ candidate_weight.T
        update = (residual.T @ train_quantized) / denominator.clamp_min(
            torch.finfo(torch.float32).tiny
        ).unsqueeze(0)
        continuous_weight = candidate_weight + _WEIGHT_REFINEMENT_STEP * update
        signed_code = torch.round(continuous_weight / scalar_scale * 4.0).clamp(
            -7, 7
        )
        shaped_code = signed_code.reshape_as(weight_params["mant"])
        candidate["sign"] = shaped_code.sign().to(torch.bfloat16)
        candidate["mant"] = (shaped_code.abs() / 4.0).to(torch.bfloat16)
        if iteration in _WEIGHT_REFINEMENT_CHECKPOINTS:
            checkpoints.append(dict(candidate))
    checkpoints.append(
        _quantize_values(continuous_weight, train_quantized.square().mean(0))
    )

    heldout_target = activation_parts[-1] @ weight.T
    heldout_quantized = quantized_parts[-1]
    return _choose_weight_rows(heldout_target, heldout_quantized, checkpoints)


def _rms(values: torch.Tensor) -> torch.Tensor:
    return values.to(torch.float32).reshape(-1, values.shape[-1]).square().mean(0).sqrt()


def _balanced_multiplier(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    tiny = torch.finfo(torch.float32).tiny
    multiplier = (right.clamp_min(tiny) / left.clamp_min(tiny)).sqrt()
    multiplier = multiplier.clamp(1.0 / 16.0, 16.0)
    return torch.where((left > 0) & (right > 0), multiplier, torch.ones_like(multiplier))


def _linear_multiplier(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    tiny = torch.finfo(torch.float32).tiny
    multiplier = (
        right.clamp_min(tiny) / left.clamp_min(tiny)
    ).pow(_LINEAR_BALANCE_EXPONENT)
    multiplier = multiplier.clamp(1.0 / 16.0, 16.0)
    return torch.where((left > 0) & (right > 0), multiplier, torch.ones_like(multiplier))


_HADAMARD_CACHE: dict[int, torch.Tensor] = {}


def _hadamard_matrix(block: int) -> torch.Tensor:
    cached = _HADAMARD_CACHE.get(block)
    if cached is not None:
        return cached
    if block < 2 or block & (block - 1):
        return torch.empty(0)
    base = torch.ones(1, 1)
    size = 1
    while size < block:
        top = torch.cat([base, base], dim=1)
        bot = torch.cat([base, -base], dim=1)
        base = torch.cat([top, bot], dim=0)
        size *= 2
    matrix = (base / (block**0.5)).to(torch.float32)
    _HADAMARD_CACHE[block] = matrix
    return matrix


def _hadamard(values: torch.Tensor, block: int) -> torch.Tensor:
    matrix = _hadamard_matrix(block)
    if matrix.numel() == 0:
        result = values.to(torch.float32).reshape(
            -1, values.shape[-1] // block, block
        ).clone()
        stride = 1
        while stride < block:
            pairs = result.reshape(
                *result.shape[:-1], block // (2 * stride), 2, stride
            )
            left = pairs[..., 0, :].clone()
            right = pairs[..., 1, :].clone()
            pairs[..., 0, :] = left + right
            pairs[..., 1, :] = left - right
            stride *= 2
        return (result / block**0.5).reshape_as(values)
    flat = values.to(torch.float32).reshape(-1, block)
    return (flat @ matrix).reshape_as(values)


def _sign_candidates() -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260819)
    random = torch.randint(0, 2, (7, 64), generator=generator).to(torch.float32)
    return torch.cat((torch.ones(1, 64), random.mul_(2).sub_(1)))


def _peak_ratio(values: torch.Tensor) -> torch.Tensor:
    peak = values.abs().amax(-1).square().sum(0)
    energy = values.square().sum(dim=(0, 2))
    return peak / energy.clamp_min(torch.finfo(torch.float32).tiny)


def _select_linear_signs(activation: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    group_count = activation.shape[-1] // 64
    a = activation.reshape(-1, group_count, 64)
    w = weight.reshape(-1, group_count, 64)
    candidates = _sign_candidates()
    scores = [
        _peak_ratio(_hadamard(a * signs, 64)) + _peak_ratio(_hadamard(w * signs, 64))
        for signs in candidates
    ]
    return candidates[torch.stack(scores).argmin(0)].reshape(-1)


def _sample_sign_weight_rows(weight: torch.Tensor) -> torch.Tensor:
    if weight.shape[0] <= _LINEAR_SIGN_WEIGHT_ROWS:
        return weight
    indices = torch.linspace(
        0,
        weight.shape[0] - 1,
        _LINEAR_SIGN_WEIGHT_ROWS,
        device=weight.device,
    ).round().to(torch.int64)
    return weight.index_select(0, indices)


def _linear_transform(values: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
    return _hadamard(values.to(torch.float32) * state["coefficient"], state["hadamard"])


@torch.inference_mode()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor, weight_scale: torch.Tensor, calib_activation_list: list
) -> dict[str, Any]:
    weight = dequantize_nvfp4(weight_quant, weight_scale).to(torch.float32)
    activation_parts = [
        dequantize_nvfp4(quant, scale).to(torch.float32)
        for quant, scale in calib_activation_list
    ]
    activation = torch.cat(activation_parts)
    multiplier = _linear_multiplier(_rms(activation), _rms(weight))
    transformed_activation = activation * multiplier
    transformed_weight = weight / multiplier
    signs = _select_linear_signs(
        transformed_activation, _sample_sign_weight_rows(transformed_weight)
    )
    coefficient = multiplier * signs
    transformed_activation = _hadamard(transformed_activation * signs, _LINEAR_HADAMARD)
    transformed_weight = _hadamard(transformed_weight * signs, _LINEAR_HADAMARD)
    activation_state = {
        "version": 224,
        "coefficient": coefficient,
        "hadamard": _LINEAR_HADAMARD,
    }
    transformed_activation_parts = [
        _linear_transform(part, activation_state) for part in activation_parts
    ]
    if transformed_activation_parts:
        quantized_parts = [
            _dequantize_hif4(_quantize_values(part))
            for part in transformed_activation_parts
        ]
        all_quantized = torch.cat(quantized_parts)
        weight_params = _block_gptq_weight(
            all_quantized, transformed_weight
        )
    else:
        importance = transformed_activation.square().mean(0)
        weight_params = _quantize_values(transformed_weight, importance)
    return {"weight_params": weight_params, "activation_state": activation_state}


@torch.inference_mode()
def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor, activation_scale: torch.Tensor, activation_state: Any
) -> dict[str, torch.Tensor]:
    values = dequantize_nvfp4(activation_quant, activation_scale)
    return _quantize_values_vec(_linear_transform(values, activation_state))


# =============================================================================
# Attention
# =============================================================================

def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """Build reciprocal per-KV-head Q/K balance from calibration moments."""
    q_values = torch.cat([
        dequantize_nvfp4(*sample["q"]).to(torch.float32)
        for sample in calib_qkv_list
    ])
    k_values = torch.cat([
        dequantize_nvfp4(*sample["k"]).to(torch.float32)
        for sample in calib_qkv_list
    ])
    replicas = q_num_heads // kv_num_heads
    q_energy = q_values.reshape(-1, q_num_heads, head_dim).square().mean(0)
    grouped_q_energy = q_energy.reshape(kv_num_heads, replicas, head_dim)
    q_rms = grouped_q_energy.sqrt().mean(1)
    k_rms = k_values.reshape(-1, kv_num_heads, head_dim).square().mean(0).sqrt()
    tiny = torch.finfo(torch.float32).tiny
    balance = (k_rms.clamp_min(tiny) / q_rms.clamp_min(tiny)).sqrt()
    balance = balance.clamp(1.0 / 16.0, 16.0)
    balance = torch.where(
        (q_rms > 0) & (k_rms > 0), balance, torch.ones_like(balance)
    )
    hadamard = 1
    global_rms = k_values.square().mean().sqrt()
    if bool(global_rms > 0):
        near_zero = (
            k_values.abs() < global_rms * _ATTENTION_NEAR_ZERO_RATIO
        ).to(torch.float32).mean()
        if bool(near_zero <= _ATTENTION_SPARSE_THRESHOLD):
            hadamard = head_dim
    return {
        "q_state": {
            "coefficient": balance.repeat_interleave(replicas, dim=0),
            "hadamard": hadamard,
        },
        "k_state": {
            "coefficient": balance.reciprocal(),
            "hadamard": hadamard,
        },
        "v_state": None,
    }


def _quantize_attention_values(
    values: torch.Tensor,
    offsets: tuple[int, ...],
    energy_penalty: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Select legal scales with an optional per-block energy drift cost."""
    candidates = [
        _quantize_values_vec(values, offsets=(offset,))
        for offset in offsets
    ]
    if len(candidates) == 1:
        return candidates[0]
    channels = int(values.shape[-1])
    target = values.to(torch.float32).reshape(
        *values.shape[:-1], channels // 64, 64
    )
    target_energy = target.square().sum(-1).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    scores = []
    for params in candidates:
        reconstructed = _dequantize_hif4(params).reshape_as(target)
        residual_cost = (target - reconstructed).square().sum(-1)
        if energy_penalty:
            energy_drift = target_energy - reconstructed.square().sum(-1)
            residual_cost = residual_cost + (
                energy_penalty * energy_drift.square() / target_energy
            )
        scores.append(residual_cost)
    best = torch.stack(scores).argmin(0)
    result: dict[str, torch.Tensor] = {}
    for name in candidates[0]:
        stacked = torch.stack([candidate[name] for candidate in candidates])
        tail = stacked.shape[1 + best.ndim:]
        index = best.reshape((1,) + best.shape + (1,) * len(tail))
        index = index.expand((1,) + best.shape + tail)
        result[name] = torch.gather(stacked, 0, index).squeeze(0)
    return result


def _transform_attention_qk(
    values: torch.Tensor,
    heads: int,
    head_dim: int,
    state: dict[str, Any],
) -> torch.Tensor:
    shaped = values.reshape(-1, heads, head_dim)
    coefficient = state["coefficient"].to(values.device, values.dtype)
    shaped = shaped * coefficient.unsqueeze(0)
    return _hadamard(shaped, int(state["hadamard"])).reshape_as(values)


def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    value = dequantize_nvfp4(q_quant, q_scale).to(torch.float32)
    value = _transform_attention_qk(value, q_num_heads, head_dim, q_state)
    return _quantize_attention_values(
        value,
        _ATTENTION_QK_OFFSETS,
        _ATTENTION_ENERGY_PENALTY,
    )


def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    value = dequantize_nvfp4(k_quant, k_scale).to(torch.float32)
    value = _transform_attention_qk(value, kv_num_heads, head_dim, k_state)
    return _quantize_attention_values(
        value,
        _ATTENTION_QK_OFFSETS,
        _ATTENTION_ENERGY_PENALTY,
    )


def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    value = dequantize_nvfp4(v_quant, v_scale).to(torch.float32)
    return _quantize_attention_values(value, _ATTENTION_V_OFFSETS)
