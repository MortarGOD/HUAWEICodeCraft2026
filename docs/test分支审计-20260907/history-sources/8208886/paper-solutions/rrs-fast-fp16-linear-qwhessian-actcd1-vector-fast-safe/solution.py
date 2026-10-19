from __future__ import annotations

import math
from typing import Any

import torch


_MIN_SCALE_FACTOR = 2.0**-48
_MAX_SCALE_FACTOR = 49152.0
_MANTISSA_MAX = 1.75
_MANTISSA_STEP = 0.25
_BASE_RADIUS = 1
_ENHANCED_RADIUS = 3
_LINEAR_RADIUS = 4
_SWITCH_MEAN_GAIN = 0.001
_SWITCH_WORST_GAIN = -0.1
_IMPORTANCE_MIN = 0.25
_IMPORTANCE_MAX = 4.0
_ATTENTION_TRANSFORM_MEAN_GAIN = 0.03
_ATTENTION_TRANSFORM_WORST_GAIN = -0.05
_ATTENTION_TRANSFORM_MAX_LOG_SCALE_STD = 2.0
_ATTENTION_PERMUTED_TRANSFORM_MEAN_GAIN = 0.1
_ATTENTION_PERMUTED_TRANSFORM_WORST_GAIN = 0.0
_ATTENTION_BALANCED_TRANSFORM_MEAN_GAIN = 0.0
_ATTENTION_BALANCED_TRANSFORM_WORST_GAIN = 0.0
_ATTENTION_BALANCE_LOG2_LIMIT = 2.0
_ATTENTION_BALANCE_EXPONENT = 0.25
_LINEAR_SMOOTH_ALPHA = 0.5
_LINEAR_SMOOTH_QUANTILE_ALPHA = 0.10
_LINEAR_SMOOTH_QUANTILE = 0.977
_SCALE_LEARNING_RATE = 0.06
_SCALE_GROUP_WIDTH = 16
_SCALE_LEARNING_LOG2_LIMIT = 3.0
_LINEAR_SMOOTH_LOG2_LIMIT = 3.0
_LINEAR_SMOOTH_CENTER_WIDTH = 64
_STABILITY_STANDARD_DEVIATIONS = 3.0


def _dequantize_src(quant: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    blocked = quant.detach().to(device="cpu").unflatten(-1, (-1, 16))
    value = blocked * scale.detach().to(device="cpu").unsqueeze(-1)
    return value.flatten(-2, -1).to(torch.bfloat16).to(torch.float32).contiguous()


def _fwht64(value: torch.Tensor) -> torch.Tensor:
    if value.ndim < 1 or value.shape[-1] % 64 != 0:
        raise ValueError("Hadamard transform requires a width divisible by 64")
    original_shape = tuple(value.shape)
    transformed = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    transformed = transformed.reshape(original_shape[:-1] + (-1, 64))
    for half_width in (1, 2, 4, 8, 16, 32):
        groups = transformed.reshape(
            transformed.shape[:-1] + (64 // (2 * half_width), 2, half_width)
        )
        left = groups[..., 0, :]
        right = groups[..., 1, :]
        transformed = torch.stack((left + right, left - right), dim=-2).reshape(
            transformed.shape
        )
    return (transformed * 0.125).reshape(original_shape).contiguous()


def _apply_linear_family(
    value: torch.Tensor,
    family: int,
    permutation: torch.Tensor,
    smooth_balance: torch.Tensor,
    *,
    inverse_smooth: bool = False,
) -> torch.Tensor:
    if family == 0:
        return value
    if family == 1:
        return _fwht64(value)
    if family == 2:
        return value.index_select(-1, permutation)
    if family == 3:
        factor = (
            smooth_balance.reciprocal()
            if inverse_smooth
            else smooth_balance
        )
        return value * factor
    if family == 4:
        factor = (
            smooth_balance.reciprocal()
            if inverse_smooth
            else smooth_balance
        )
        return _fwht64(value * factor)
    raise ValueError("unknown Linear transform family")


def _linear_smooth_balance(
    weight: torch.Tensor,
    activation_energy: torch.Tensor,
    activation_count: int,
) -> torch.Tensor:
    weight_rms = torch.mean(weight.square(), dim=0).clamp_min(1.0e-24).sqrt()
    activation_rms = (
        activation_energy / max(activation_count, 1)
    ).clamp_min(1.0e-24).sqrt()
    log_balance = (
        (1.0 - _LINEAR_SMOOTH_ALPHA) * torch.log2(weight_rms)
        - _LINEAR_SMOOTH_ALPHA * torch.log2(activation_rms)
    )
    center_width = math.gcd(int(log_balance.numel()), _LINEAR_SMOOTH_CENTER_WIDTH)
    log_balance = log_balance.reshape(-1, center_width)
    log_balance = log_balance - torch.mean(
        log_balance, dim=-1, keepdim=True
    )
    return torch.pow(
        2.0,
        torch.clamp(
            log_balance,
            min=-_LINEAR_SMOOTH_LOG2_LIMIT,
            max=_LINEAR_SMOOTH_LOG2_LIMIT,
        ),
    ).flatten().contiguous()


def _linear_quantile_smooth_balance(
    weight: torch.Tensor,
    activations: list[torch.Tensor],
) -> torch.Tensor:
    activation_stat = torch.quantile(
        torch.cat([value.abs() for value in activations], dim=0),
        _LINEAR_SMOOTH_QUANTILE,
        dim=0,
    ).clamp_min(1.0e-12)
    weight_stat = torch.amax(weight.abs(), dim=0).clamp_min(1.0e-12)
    alpha = _LINEAR_SMOOTH_QUANTILE_ALPHA
    log_balance = (
        (1.0 - alpha) * torch.log2(weight_stat)
        - alpha * torch.log2(activation_stat)
    )
    log_balance = log_balance.reshape(-1, _LINEAR_SMOOTH_CENTER_WIDTH)
    log_balance = log_balance - torch.mean(
        log_balance, dim=-1, keepdim=True
    )
    return torch.pow(
        2.0,
        torch.clamp(
            log_balance,
            min=-_LINEAR_SMOOTH_LOG2_LIMIT,
            max=_LINEAR_SMOOTH_LOG2_LIMIT,
        ),
    ).flatten().contiguous()


def _learn_smooth_balance(
    weight: torch.Tensor,
    activations: list[torch.Tensor],
    references: list[torch.Tensor],
    initial_balance: torch.Tensor,
) -> torch.Tensor:
    activation = torch.cat(activations, dim=0).to(torch.float32)
    reference = torch.cat(references, dim=0).to(torch.float32)
    observed_count = torch.sum(activation != 0.0, dim=0)
    reliable = torch.sum(
        (observed_count >= 2).reshape(-1, _SCALE_GROUP_WIDTH), dim=-1
    ) >= (_SCALE_GROUP_WIDTH // 2)
    initial_log = torch.log2(initial_balance.clamp_min(2.0**-16))
    reference_energy = torch.mean(reference.square()).clamp_min(1.0e-12)
    activation_input = activation * initial_balance
    weight_input = weight / initial_balance
    activation_rows = int(activation_input.shape[0])
    transformed = _fwht64(torch.cat((activation_input, weight_input), dim=0))
    quantized = _dequantize_tgt(
        _quantize_hierarchical(transformed, _BASE_RADIUS),
        tuple(transformed.shape),
    )
    quantized_activation = quantized[:activation_rows]
    quantized_weight = quantized[activation_rows:]
    residual = (
        2.0
        * (quantized_activation @ quantized_weight.transpose(0, 1) - reference)
        / (reference.numel() * reference_energy)
    )
    activation_gradient = _fwht64(residual @ quantized_weight)
    weight_gradient = _fwht64(
        residual.transpose(0, 1) @ quantized_activation
    )
    channel_gradient = math.log(2.0) * (
        torch.sum(activation_gradient * activation_input, dim=0)
        - torch.sum(weight_gradient * weight_input, dim=0)
    )
    group_gradient = torch.sum(
        channel_gradient.reshape(-1, _SCALE_GROUP_WIDTH), dim=-1
    ).reshape(-1, 64 // _SCALE_GROUP_WIDTH)
    group_gradient = group_gradient - torch.mean(
        group_gradient, dim=-1, keepdim=True
    )
    group_gradient = group_gradient.flatten() * reliable.to(torch.float32)
    delta = -_SCALE_LEARNING_RATE * group_gradient / (
        torch.abs(group_gradient) + 1.0e-8
    )
    learned_delta = delta.repeat_interleave(_SCALE_GROUP_WIDTH).reshape(-1, 64)
    learned_delta = learned_delta - torch.mean(
        learned_delta, dim=-1, keepdim=True
    )
    return torch.pow(
        2.0,
        torch.clamp(
            initial_log + learned_delta.flatten(),
            min=-_SCALE_LEARNING_LOG2_LIMIT,
            max=_SCALE_LEARNING_LOG2_LIMIT,
        ),
    ).contiguous()


def _apply_exact_scale_gradient(
    initial_balance: torch.Tensor,
    channel_gradient: torch.Tensor,
    learning_rate: float,
) -> torch.Tensor:
    group_gradient = torch.sum(
        channel_gradient.reshape(-1, _SCALE_GROUP_WIDTH), dim=-1
    ).reshape(-1, 64 // _SCALE_GROUP_WIDTH)
    group_gradient = group_gradient - torch.mean(
        group_gradient, dim=-1, keepdim=True
    )
    delta = -learning_rate * group_gradient.flatten() / (
        torch.abs(group_gradient.flatten()) + 1.0e-8
    )
    learned_delta = delta.repeat_interleave(_SCALE_GROUP_WIDTH).reshape(-1, 64)
    learned_delta = learned_delta - torch.mean(
        learned_delta, dim=-1, keepdim=True
    )
    initial_log = torch.log2(initial_balance.clamp_min(2.0**-16))
    return torch.pow(
        2.0,
        torch.clamp(
            initial_log + learned_delta.flatten(),
            min=-_SCALE_LEARNING_LOG2_LIMIT,
            max=_SCALE_LEARNING_LOG2_LIMIT,
        ),
    ).contiguous()


def _predicted_scale_descent(channel_gradient: torch.Tensor) -> float:
    group_gradient = torch.sum(
        channel_gradient.reshape(-1, _SCALE_GROUP_WIDTH), dim=-1
    ).reshape(-1, 64 // _SCALE_GROUP_WIDTH)
    group_gradient = group_gradient - torch.mean(
        group_gradient, dim=-1, keepdim=True
    )
    return float(
        (_SCALE_LEARNING_RATE * torch.sum(torch.abs(group_gradient))).item()
    )


def _probe_scale_pair(
    weight: torch.Tensor,
    activations: list[torch.Tensor],
    references: list[torch.Tensor],
    balances: tuple[torch.Tensor, torch.Tensor],
) -> tuple[list[float], list[float]]:
    transformed_weights = torch.stack(
        [_fwht64(weight / balance) for balance in balances], dim=0
    )
    weight_values = _dequantize_tgt(
        _quantize_hierarchical(transformed_weights, _BASE_RADIUS),
        tuple(transformed_weights.shape),
    )
    transformed_activations = torch.stack(
        [
            torch.stack(
                [_fwht64(activation * balance) for activation in activations],
                dim=0,
            )
            for balance in balances
        ],
        dim=0,
    )
    activation_values = _dequantize_tgt(
        _quantize_hierarchical(transformed_activations, _BASE_RADIUS),
        tuple(transformed_activations.shape),
    )
    outputs = torch.einsum(
        "rsnc,roc->rsno", activation_values, weight_values
    )
    target = torch.stack(references, dim=0).unsqueeze(0)
    losses = torch.mean(
        (outputs.to(torch.float32) - target.to(torch.float32)).square(),
        dim=(-2, -1),
    )
    return (
        [float(value.item()) for value in losses[0]],
        [float(value.item()) for value in losses[1]],
    )


def _linear_channel_stability(activations: list[torch.Tensor]) -> float:
    if len(activations) < 2:
        return -1.0
    patterns = []
    for activation in activations:
        log_energy = torch.log2(
            torch.mean(activation.square(), dim=0).clamp_min(1.0e-24)
        )
        center_width = math.gcd(
            int(log_energy.numel()), _LINEAR_SMOOTH_CENTER_WIDTH
        )
        blocks = log_energy.reshape(-1, center_width)
        pattern = (blocks - torch.mean(blocks, dim=-1, keepdim=True)).flatten()
        patterns.append(
            pattern / torch.linalg.vector_norm(pattern).clamp_min(1.0e-12)
        )
    pattern_matrix = torch.stack(patterns)
    pattern_count = len(patterns)
    similarities = pattern_matrix @ pattern_matrix.transpose(0, 1)
    return float(
        (
            (torch.sum(similarities) - pattern_count)
            / (pattern_count * (pattern_count - 1))
        ).item()
    )


def _stability_threshold(feature_count: int) -> float:
    return _STABILITY_STANDARD_DEVIATIONS / math.sqrt(max(feature_count, 1))


def _e6m2_candidates(block_peak: torch.Tensor, radius: int) -> torch.Tensor:
    target = torch.clamp(
        block_peak.to(torch.float64) / (4.0 * _MANTISSA_MAX),
        min=_MIN_SCALE_FACTOR,
        max=_MAX_SCALE_FACTOR,
    )
    exponent = torch.floor(torch.log2(target))
    significand_index = torch.ceil(target / torch.pow(2.0, exponent) * 4.0) - 4.0
    integer_index = significand_index.to(torch.int64)
    carry = torch.div(integer_index, 4, rounding_mode="floor")
    base_rank = (exponent.to(torch.int64) + carry) * 4 + torch.remainder(
        integer_index, 4
    )
    offsets = torch.arange(
        -radius,
        radius + 1,
        dtype=torch.int64,
        device=block_peak.device,
    )
    ranks = base_rank.unsqueeze(-1) + offsets
    candidate_exponents = torch.div(ranks, 4, rounding_mode="floor")
    candidate_mantissas = 4 + torch.remainder(ranks, 4)
    candidates = candidate_mantissas.to(torch.float64) * torch.pow(
        2.0, candidate_exponents.to(torch.float64) - 2.0
    )
    return torch.clamp(
        candidates, min=_MIN_SCALE_FACTOR, max=_MAX_SCALE_FACTOR
    ).to(torch.float32)


def _quantize_multiplier(
    x: torch.Tensor,
    absolute: torch.Tensor,
    input_sign: torch.Tensor,
    scale_factor: torch.Tensor,
    multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    effective_scale = scale_factor * multiplier
    mantissa = torch.clamp(
        torch.round(absolute / effective_scale / _MANTISSA_STEP)
        * _MANTISSA_STEP,
        min=0.0,
        max=_MANTISSA_MAX,
    )
    reconstructed = input_sign * mantissa * effective_scale
    return mantissa, (reconstructed - x).square()


def _reduce_quartet_error(
    squared_error: torch.Tensor,
    weight: torch.Tensor | None,
) -> torch.Tensor:
    if weight is None:
        return torch.sum(squared_error, dim=-1, keepdim=True)
    return torch.sum(weight * squared_error, dim=-1, keepdim=True)


def _select_at_scale(
    mantissas: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    quartet_errors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mantissa_1, mantissa_2, mantissa_4 = mantissas
    error_1, error_2, error_4 = quartet_errors

    use_high_lv3_for_low_lv2 = error_2 < error_1
    low_lv2_lv3 = torch.where(use_high_lv3_for_low_lv2, 2.0, 1.0).to(
        torch.float32
    )
    low_lv2_mantissa = torch.where(
        use_high_lv3_for_low_lv2, mantissa_2, mantissa_1
    )
    low_lv2_quartet_error = torch.where(
        use_high_lv3_for_low_lv2, error_2, error_1
    )
    low_lv2_group_error = torch.sum(
        low_lv2_quartet_error, dim=-2, keepdim=True
    )

    use_high_lv3_for_high_lv2 = error_4 < error_2
    high_lv2_lv3 = torch.where(use_high_lv3_for_high_lv2, 2.0, 1.0).to(
        torch.float32
    )
    high_lv2_mantissa = torch.where(
        use_high_lv3_for_high_lv2, mantissa_4, mantissa_2
    )
    high_lv2_quartet_error = torch.where(
        use_high_lv3_for_high_lv2, error_4, error_2
    )
    high_lv2_group_error = torch.sum(
        high_lv2_quartet_error, dim=-2, keepdim=True
    )

    use_high_lv2 = high_lv2_group_error < low_lv2_group_error
    group_error = torch.where(
        use_high_lv2, high_lv2_group_error, low_lv2_group_error
    )
    lv2 = torch.where(use_high_lv2, 2.0, 1.0).to(torch.float32)
    lv3 = torch.where(use_high_lv2, high_lv2_lv3, low_lv2_lv3)
    mantissa = torch.where(
        use_high_lv2, high_lv2_mantissa, low_lv2_mantissa
    )
    block_error = torch.sum(group_error, dim=(-3, -2, -1), keepdim=True)
    return block_error, lv2, lv3, mantissa


def _expand_importance(
    importance: torch.Tensor | None,
    original_shape: tuple[int, ...],
    blocked_shape: tuple[int, ...],
) -> torch.Tensor | None:
    if importance is None:
        return None
    source = importance.detach().to(device="cpu", dtype=torch.float32)
    if source.ndim == 1 and source.numel() == original_shape[-1]:
        if not bool(torch.isfinite(source).all().item()) or bool(
            (source < 0).any().item()
        ):
            raise ValueError("importance must contain finite non-negative values")
        view_shape = (1,) * (len(original_shape) - 1) + blocked_shape[-4:]
        return source.reshape(view_shape).expand(blocked_shape)
    try:
        expanded = torch.broadcast_to(source, original_shape)
    except RuntimeError as error:
        raise ValueError("importance cannot be broadcast to the input shape") from error
    if not bool(torch.isfinite(expanded).all().item()) or bool(
        (expanded < 0).any().item()
    ):
        raise ValueError("importance must contain finite non-negative values")
    return expanded.contiguous().reshape(blocked_shape)


def _prepare_block_quadratic(
    quadratic: torch.Tensor | None,
    channels: int,
) -> torch.Tensor | None:
    if quadratic is None:
        return None
    value = quadratic.detach().to(device="cpu", dtype=torch.float32)
    expected_shape = (channels // 64, 64, 64)
    if tuple(value.shape) != expected_shape:
        raise ValueError("quadratic must contain one 64x64 matrix per block")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError("quadratic must contain finite values")
    return value.contiguous()


def _quadratic_block_error(
    error: torch.Tensor,
    quadratic: torch.Tensor,
) -> torch.Tensor:
    vectors = error.flatten(-3, -1)
    block_error = torch.einsum(
        "...bi,bij,...bj->...b", vectors, quadratic, vectors
    )
    return block_error[..., None, None, None]


def _update_best_candidate(
    best: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | None,
    error: torch.Tensor,
    scale_factor: torch.Tensor,
    lv2: torch.Tensor,
    lv3: torch.Tensor,
    mantissa: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if best is None:
        return error, scale_factor, lv2, lv3, mantissa
    best_error, best_scale, best_lv2, best_lv3, best_mantissa = best
    use_candidate = error < best_error
    return (
        torch.where(use_candidate, error, best_error),
        torch.where(use_candidate, scale_factor, best_scale),
        torch.where(use_candidate, lv2, best_lv2),
        torch.where(use_candidate, lv3, best_lv3),
        torch.where(use_candidate, mantissa, best_mantissa),
    )


def _finalize_candidate(
    best: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    input_sign: torch.Tensor,
) -> dict[str, torch.Tensor]:
    _, scale_factor, lv2, lv3, mantissa = best
    sign = torch.where(mantissa == 0.0, 0.0, input_sign)
    return {
        "scale_factor": scale_factor.contiguous(),
        "scale_lv2": lv2.contiguous(),
        "scale_lv3": lv3.contiguous(),
        "sign": sign.contiguous(),
        "mant": mantissa.contiguous(),
    }


def _refine_quadratic_groups(
    x: torch.Tensor,
    quadratic: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    refined = {name: value.clone() for name, value in params.items()}
    scale = refined["scale_factor"].reshape(x.shape[:-3] + (1,))
    reconstructed = (
        refined["sign"]
        * refined["mant"]
        * refined["scale_lv2"]
        * refined["scale_lv3"]
        * refined["scale_factor"]
    )
    flat_error = (reconstructed - x).flatten(-3, -1)
    lv2_options = torch.tensor(
        [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
        dtype=torch.float32,
    )
    lv3_options = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [2.0, 2.0],
        ],
        dtype=torch.float32,
    )
    multipliers = (
        lv2_options[:, None, None] * lv3_options[:, :, None]
    ).expand(8, 2, 4)

    effective_scale = (
        scale[..., 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        * multipliers
    )
    candidate_mantissa = torch.clamp(
        torch.round(
            torch.abs(x).unsqueeze(-3)
            / effective_scale
            / _MANTISSA_STEP
        )
        * _MANTISSA_STEP,
        min=0.0,
        max=_MANTISSA_MAX,
    )
    candidate_value = (
        torch.sign(x).unsqueeze(-3)
        * candidate_mantissa
        * effective_scale
    )
    current_value = x + flat_error.reshape_as(x)
    delta = (
        candidate_value - current_value.unsqueeze(-3)
    ).flatten(-2, -1)
    gradient = torch.einsum(
        "...bi,bij->...bj", flat_error, quadratic
    ).reshape(x.shape[:-3] + (8, 8))
    group_quadratic = torch.stack(
        [
            quadratic[:, index * 8 : (index + 1) * 8,
                      index * 8 : (index + 1) * 8]
            for index in range(8)
        ],
        dim=1,
    )
    delta_objective = 2.0 * torch.sum(
        delta * gradient.unsqueeze(-2), dim=-1
    ) + torch.einsum(
        "...bgki,bgij,...bgkj->...bgk",
        delta,
        group_quadratic,
        delta,
    )
    best_delta, best_index = torch.min(delta_objective, dim=-1)
    global_delta, best_group = torch.min(best_delta, dim=-1)
    improve = global_delta < 0.0
    group_ids = torch.arange(8, dtype=torch.int64).reshape(
        (1,) * best_group.ndim + (8,)
    )
    update_group = (
        best_group.unsqueeze(-1) == group_ids
    ) & improve.unsqueeze(-1)

    gather_index = best_index[..., None, None, None].expand(
        best_index.shape + (1, 2, 4)
    )
    best_mantissa = torch.gather(
        candidate_mantissa, -3, gather_index
    ).squeeze(-3)
    refined["mant"] = torch.where(
        update_group[..., None, None], best_mantissa, refined["mant"]
    )
    refined["sign"] = torch.where(
        refined["mant"] == 0.0, 0.0, torch.sign(x)
    )
    selected_lv2 = lv2_options[best_index]
    refined["scale_lv2"] = torch.where(
        update_group[..., None, None],
        selected_lv2[..., None, None],
        refined["scale_lv2"],
    )
    selected_lv3 = lv3_options[best_index]
    refined["scale_lv3"] = torch.where(
        update_group[..., None, None],
        selected_lv3[..., None],
        refined["scale_lv3"],
    )
    return {name: value.contiguous() for name, value in refined.items()}


def _quantize_hierarchical_multi(
    value: torch.Tensor,
    unweighted_radii: tuple[int, ...],
    weighted_radius: int | None = None,
    importance: torch.Tensor | None = None,
    quadratic: torch.Tensor | None = None,
    refine_quadratic: bool = False,
) -> dict[tuple[str, int], dict[str, torch.Tensor]]:
    if not isinstance(value, torch.Tensor) or value.ndim < 1:
        raise ValueError("value must be a non-scalar tensor")
    if not unweighted_radii and weighted_radius is None:
        raise ValueError("at least one quantization route is required")
    if weighted_radius is not None and importance is None:
        raise ValueError("weighted quantization requires importance")
    if quadratic is not None and weighted_radius is None:
        raise ValueError("quadratic requires weighted quantization")
    if refine_quadratic and quadratic is None:
        raise ValueError("refinement requires quadratic")
    requested_radii = list(unweighted_radii)
    if weighted_radius is not None:
        requested_radii.append(weighted_radius)
    if any(type(radius) is not int or radius < 0 for radius in requested_radii):
        raise ValueError("radius must be a non-negative int")

    channels = int(value.shape[-1])
    if channels % 64 != 0:
        raise ValueError("the last dimension must be divisible by 64")
    original_shape = tuple(value.shape)
    blocked_shape = original_shape[:-1] + (channels // 64, 8, 2, 4)
    x = value.detach().to(device="cpu", dtype=torch.float32).contiguous().reshape(
        blocked_shape
    )
    if not bool(torch.isfinite(x).all().item()):
        raise ValueError("value contains NaN or infinity")

    weight = _expand_importance(importance, original_shape, blocked_shape)
    block_quadratic = _prepare_block_quadratic(quadratic, channels)
    absolute = torch.abs(x)
    input_sign = torch.sign(x)
    block_peak = torch.amax(absolute, dim=(-3, -2, -1), keepdim=True)
    maximum_radius = max(requested_radii)
    candidates = _e6m2_candidates(block_peak, maximum_radius)
    unweighted_best = {radius: None for radius in unweighted_radii}
    weighted_best = None

    for candidate_index in range(candidates.shape[-1]):
        offset = candidate_index - maximum_radius
        scale_factor = candidates[..., candidate_index]
        mantissa_1, squared_error_1 = _quantize_multiplier(
            x, absolute, input_sign, scale_factor, 1.0
        )
        mantissa_2, squared_error_2 = _quantize_multiplier(
            x, absolute, input_sign, scale_factor, 2.0
        )
        mantissa_4, squared_error_4 = _quantize_multiplier(
            x, absolute, input_sign, scale_factor, 4.0
        )
        mantissas = (mantissa_1, mantissa_2, mantissa_4)

        active_unweighted = [
            radius for radius in unweighted_radii if abs(offset) <= radius
        ]
        if active_unweighted:
            unweighted_errors = (
                _reduce_quartet_error(squared_error_1, None),
                _reduce_quartet_error(squared_error_2, None),
                _reduce_quartet_error(squared_error_4, None),
            )
            error, lv2, lv3, mantissa = _select_at_scale(
                mantissas, unweighted_errors
            )
            for radius in active_unweighted:
                unweighted_best[radius] = _update_best_candidate(
                    unweighted_best[radius],
                    error,
                    scale_factor,
                    lv2,
                    lv3,
                    mantissa,
                )

        if weighted_radius is not None and abs(offset) <= weighted_radius:
            weighted_errors = (
                _reduce_quartet_error(squared_error_1, weight),
                _reduce_quartet_error(squared_error_2, weight),
                _reduce_quartet_error(squared_error_4, weight),
            )
            error, lv2, lv3, mantissa = _select_at_scale(
                mantissas, weighted_errors
            )
            if block_quadratic is not None:
                reconstructed = (
                    input_sign
                    * mantissa
                    * scale_factor
                    * lv2
                    * lv3
                )
                error = _quadratic_block_error(
                    reconstructed - x, block_quadratic
                )
            weighted_best = _update_best_candidate(
                weighted_best,
                error,
                scale_factor,
                lv2,
                lv3,
                mantissa,
            )

    results = {
        ("unweighted", radius): _finalize_candidate(best, input_sign)
        for radius, best in unweighted_best.items()
    }
    if weighted_radius is not None:
        weighted_params = _finalize_candidate(
            weighted_best, input_sign
        )
        if refine_quadratic:
            weighted_params = _refine_quadratic_groups(
                x, block_quadratic, weighted_params
            )
        results[("weighted", weighted_radius)] = weighted_params
    return results


def _quantize_hierarchical(
    value: torch.Tensor,
    radius: int,
    importance: torch.Tensor | None = None,
    quadratic: torch.Tensor | None = None,
    refine_quadratic: bool = False,
) -> dict[str, torch.Tensor]:
    if importance is None:
        return _quantize_hierarchical_multi(value, (radius,))[
            ("unweighted", radius)
        ]
    return _quantize_hierarchical_multi(
        value,
        (),
        weighted_radius=radius,
        importance=importance,
        quadratic=quadratic,
        refine_quadratic=refine_quadratic,
    )[("weighted", radius)]


def _dequantize_tgt(
    params: dict[str, torch.Tensor], original_shape: tuple[int, ...]
) -> torch.Tensor:
    value = (
        params["sign"].to(torch.float32)
        * params["mant"].to(torch.float32)
        * params["scale_lv2"].to(torch.float32)
        * params["scale_lv3"].to(torch.float32)
        * params["scale_factor"].to(torch.float32)
    )
    return value.reshape(original_shape)


def _normalize_importance(importance: torch.Tensor) -> torch.Tensor:
    flat = importance.detach().to(device="cpu", dtype=torch.float32).flatten()
    if flat.numel() % 64 != 0:
        raise ValueError("importance width must be divisible by 64")
    blocks = flat.reshape(-1, 64)
    means = torch.mean(blocks, dim=-1, keepdim=True).clamp_min(1.0e-12)
    normalized = blocks / means
    return torch.clamp(
        normalized, min=_IMPORTANCE_MIN, max=_IMPORTANCE_MAX
    ).flatten().contiguous()


def _relative_gains(base_losses: list[float], candidate_losses: list[float]) -> list[float]:
    return [
        (base - candidate) / max(base, 1.0e-12)
        for base, candidate in zip(base_losses, candidate_losses)
    ]


def _select_candidate(losses: list[list[float]]) -> int:
    best_index = 0
    best_mean = 0.0
    base_losses = losses[0]
    for index in range(1, len(losses)):
        gains = _relative_gains(base_losses, losses[index])
        mean_gain = sum(gains) / len(gains)
        if (
            mean_gain >= _SWITCH_MEAN_GAIN
            and min(gains) >= _SWITCH_WORST_GAIN
            and mean_gain > best_mean
        ):
            best_index = index
            best_mean = mean_gain
    return best_index


def _use_learned_scale(
    base_losses: list[float], learned_losses: list[float]
) -> bool:
    gains = [
        (base - learned) / max(base, 1.0e-12)
        for base, learned in zip(base_losses, learned_losses)
    ]
    validation = gains[3:]
    mean_gain = sum(gains) / len(gains)
    return (
        0.03 <= mean_gain <= 0.25
        and sum(validation) / len(validation) >= 0.10
        and min(validation) >= -0.02
    )


def _select_attention_candidate(losses: list[list[float]]) -> int:
    """Select an Attention route from the calibration mean gain.

    Attention output error varies sharply between calibration samples. The
    per-sample worst-case gate used by Linear can reject a route whose mean
    calibration gain is consistently useful for the held-out samples.
    """
    best_index = 0
    best_mean = 0.0
    base_losses = losses[0]
    for index in range(1, len(losses)):
        gains = _relative_gains(base_losses, losses[index])
        mean_gain = sum(gains) / len(gains)
        if mean_gain >= _SWITCH_MEAN_GAIN and mean_gain > best_mean:
            best_index = index
            best_mean = mean_gain
    return best_index


def _mse(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.mean((left.to(torch.float32) - right.to(torch.float32)).square()).item())


def _linear_output(activation: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return activation.to(torch.float32) @ weight.to(torch.float32).transpose(0, 1)


def _block_second_moment(
    values: list[torch.Tensor],
    *,
    require_independent_samples: bool = False,
) -> torch.Tensor:
    if not values:
        raise ValueError("values must be non-empty")
    channels = int(values[0].shape[-1])
    blocks = channels // 64
    outer_sum = torch.zeros((blocks, 64, 64), dtype=torch.float32)
    outer_square_sum = torch.zeros_like(outer_sum)
    row_count = 0
    for value in values:
        rows = value.to(torch.float32).reshape(-1, blocks, 64)
        outer_sum += torch.einsum("nbi,nbj->bij", rows, rows)
        squared_rows = rows.square()
        outer_square_sum += torch.einsum(
            "nbi,nbj->bij", squared_rows, squared_rows
        )
        row_count += int(rows.shape[0])
    moment = outer_sum / max(row_count, 1)
    diagonal = torch.diag_embed(torch.diagonal(moment, dim1=-2, dim2=-1))
    if row_count < 2 or (require_independent_samples and len(values) < 2):
        return diagonal.contiguous()

    off_diagonal = moment - diagonal
    variance_of_mean = torch.clamp(
        outer_square_sum - row_count * moment.square(), min=0.0
    ) / (row_count * (row_count - 1))
    variance_of_mean = variance_of_mean - torch.diag_embed(
        torch.diagonal(variance_of_mean, dim1=-2, dim2=-1)
    )
    noise = torch.sum(variance_of_mean, dim=(-2, -1))
    observed_energy = torch.sum(off_diagonal.square(), dim=(-2, -1))
    shrinkage = torch.clamp(
        noise / observed_energy.clamp_min(1.0e-24), min=0.0, max=1.0
    )
    return (
        diagonal
        + (1.0 - shrinkage[:, None, None]) * off_diagonal
    ).contiguous()


def _attention_probabilities(
    q: torch.Tensor,
    k: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    seq_len = int(q.shape[0])
    qh = q.reshape(seq_len, q_num_heads, head_dim).transpose(0, 1).to(torch.float32)
    kh = k.reshape(seq_len, kv_num_heads, head_dim).transpose(0, 1).to(torch.float32)
    if q_num_heads != kv_num_heads:
        kh = kh.repeat_interleave(q_num_heads // kv_num_heads, dim=0)
    logits = torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(head_dim)
    return torch.softmax(logits, dim=-1, dtype=torch.float32)


def _attention_value_heads(
    v: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    seq_len = int(v.shape[0])
    vh = v.reshape(seq_len, kv_num_heads, head_dim).transpose(0, 1).to(torch.float32)
    if q_num_heads != kv_num_heads:
        vh = vh.repeat_interleave(q_num_heads // kv_num_heads, dim=0)
    return vh


def _attention_from_probabilities(
    probabilities: torch.Tensor,
    value_heads: torch.Tensor,
    seq_len: int,
    q_num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    output = torch.matmul(probabilities, value_heads)
    return output.transpose(0, 1).contiguous().reshape(seq_len, q_num_heads * head_dim)


def _attention_output(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    probabilities = _attention_probabilities(
        q, k, q_num_heads, kv_num_heads, head_dim
    )
    value_heads = _attention_value_heads(
        v, q_num_heads, kv_num_heads, head_dim
    )
    return _attention_from_probabilities(
        probabilities, value_heads, int(q.shape[0]), q_num_heads, head_dim
    )


def _quantize_src(
    quant: torch.Tensor,
    scale: torch.Tensor,
    radius: int,
    importance: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    return _quantize_hierarchical(_dequantize_src(quant, scale), radius, importance)


def _calibrate_rrs_linear_route(
    weight: torch.Tensor,
    activations: list[torch.Tensor],
    references: list[torch.Tensor],
    balance: torch.Tensor,
    *,
    loss_indices: tuple[int, ...] | None = None,
    compute_gradient: bool = True,
) -> tuple[
    dict[str, torch.Tensor], dict[str, Any], list[float], torch.Tensor
]:
    family_weight = _fwht64(weight / balance)
    family_activations = [
        _fwht64(activation * balance) for activation in activations
    ]
    activation_energy = torch.zeros(
        family_weight.shape[-1], dtype=torch.float32
    )
    sample_count = 0
    for activation in family_activations:
        activation_energy += torch.sum(activation.square(), dim=0)
        sample_count += int(activation.shape[0])
    activation_importance = _normalize_importance(
        activation_energy / max(sample_count, 1)
    )
    activation_quadratic = _block_second_moment(
        family_activations,
        require_independent_samples=True,
    )
    weight_params = _quantize_hierarchical(
        family_weight,
        _LINEAR_RADIUS,
        activation_importance,
        activation_quadratic,
    )
    weight_value = _dequantize_tgt(
        weight_params, tuple(family_weight.shape)
    )
    dynamic_importance = _normalize_importance(
        torch.sum(weight_value.square(), dim=0)
    )
    dynamic_quadratic = _block_second_moment([weight_value])
    losses = []
    channel_gradient = (
        torch.zeros(weight.shape[-1], dtype=torch.float32)
        if compute_gradient
        else torch.empty(0, dtype=torch.float32)
    )
    transformed_weight_gradient = (
        torch.zeros_like(family_weight) if compute_gradient else None
    )
    training_count = min(3, len(family_activations))
    for index, (source_activation, activation, reference) in enumerate(
        zip(activations, family_activations, references)
    ):
        if loss_indices is not None and index not in loss_indices:
            continue
        activation_params = _quantize_hierarchical(
            activation,
            _LINEAR_RADIUS,
            dynamic_importance,
            dynamic_quadratic,
            refine_quadratic=True,
        )
        activation_value = _dequantize_tgt(
            activation_params, tuple(activation.shape)
        )
        losses.append(
            _mse(_linear_output(activation_value, weight_value), reference)
        )
        if compute_gradient and index < training_count:
            reference_energy = torch.mean(reference.square()).clamp_min(1.0e-12)
            residual = (
                2.0
                * (_linear_output(activation_value, weight_value) - reference)
                / (training_count * reference.numel() * reference_energy)
            )
            activation_gradient = _fwht64(residual @ weight_value)
            channel_gradient += torch.sum(
                activation_gradient * (source_activation * balance), dim=0
            )
            transformed_weight_gradient += residual.transpose(0, 1) @ activation_value
    if compute_gradient:
        weight_gradient = _fwht64(transformed_weight_gradient)
        channel_gradient = math.log(2.0) * (
            channel_gradient
            - torch.sum(weight_gradient * (weight / balance), dim=0)
        )
    state = {
        "family": 4,
        "mode": 1,
        "radius": _LINEAR_RADIUS,
        "importance": dynamic_importance,
        "quadratic": dynamic_quadratic.to(torch.float16),
        "permutation": torch.empty(0, dtype=torch.int64),
        "smooth_balance": balance,
    }
    return weight_params, state, losses, channel_gradient


def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    if type(calib_activation_list) is not list or not calib_activation_list:
        raise ValueError("calib_activation_list must be a non-empty list")

    with torch.no_grad():
        weight_ref = _dequantize_src(weight_quant, weight_scale)
        activation_refs = [_dequantize_src(q, s) for q, s in calib_activation_list]
        raw_activation_energy = torch.zeros(
            weight_ref.shape[-1], dtype=torch.float32
        )
        raw_sample_count = 0
        for activation in activation_refs:
            raw_activation_energy += torch.sum(activation.square(), dim=0)
            raw_sample_count += int(activation.shape[0])
        smooth_balance = _linear_smooth_balance(
            weight_ref,
            raw_activation_energy,
            raw_sample_count,
        )
        enable_quadratic = len(activation_refs) >= 2
        if enable_quadratic:
            weight_params, state, _, _ = _calibrate_rrs_linear_route(
                weight_ref,
                activation_refs,
                [],
                smooth_balance,
                loss_indices=(),
                compute_gradient=False,
            )
            state["smooth_policy_route"] = 0
            state["probe_eligible"] = False
            return {
                "weight_params": weight_params,
                "activation_state": state,
            }
        reference_outputs = [
            _linear_output(activation, weight_ref) for activation in activation_refs
        ]
        permutation = torch.argsort(
            raw_activation_energy / max(raw_sample_count, 1),
            stable=True,
        ).to(torch.int64).contiguous()
        enable_smooth = (
            _linear_channel_stability(activation_refs)
            >= _stability_threshold(int(weight_ref.shape[-1]))
        )
        families = (0, 1, 2, 3, 4) if enable_smooth else (0, 1, 2)
        losses = [[] for _ in range(len(families) * 4)]
        family_weight_params = []
        family_dynamic_importance = []
        family_dynamic_quadratic = []

        for family in families:
            family_weight = _apply_linear_family(
                weight_ref,
                family,
                permutation,
                smooth_balance,
                inverse_smooth=True,
            )
            family_activations = [
                _apply_linear_family(
                    activation,
                    family,
                    permutation,
                    smooth_balance,
                )
                for activation in activation_refs
            ]
            activation_energy = torch.zeros(
                family_weight.shape[-1], dtype=torch.float32
            )
            sample_count = 0
            for activation in family_activations:
                activation_energy += torch.sum(activation.square(), dim=0)
                sample_count += int(activation.shape[0])
            activation_importance = _normalize_importance(
                activation_energy / max(sample_count, 1)
            )
            activation_quadratic = (
                _block_second_moment(
                    family_activations,
                    require_independent_samples=True,
                )
                if enable_quadratic
                else None
            )
            weight_routes = _quantize_hierarchical_multi(
                family_weight,
                (_BASE_RADIUS,),
                weighted_radius=_ENHANCED_RADIUS,
                importance=activation_importance,
                quadratic=activation_quadratic,
            )
            weight_params = [
                weight_routes[("unweighted", _BASE_RADIUS)],
                weight_routes[("weighted", _ENHANCED_RADIUS)],
            ]
            weight_values = [
                _dequantize_tgt(params, tuple(family_weight.shape))
                for params in weight_params
            ]
            dynamic_importance = _normalize_importance(
                torch.sum(weight_values[1].square(), dim=0)
            )
            dynamic_quadratic = (
                _block_second_moment([weight_values[1]])
                if enable_quadratic
                else None
            )
            family_weight_params.append(weight_params)
            family_dynamic_importance.append(dynamic_importance)
            family_dynamic_quadratic.append(dynamic_quadratic)

            for sample_index, activation in enumerate(family_activations):
                activation_routes = _quantize_hierarchical_multi(
                    activation,
                    (_BASE_RADIUS,),
                    weighted_radius=_ENHANCED_RADIUS,
                    importance=dynamic_importance,
                    quadratic=dynamic_quadratic,
                )
                activation_params = [
                    activation_routes[("unweighted", _BASE_RADIUS)],
                    activation_routes[("weighted", _ENHANCED_RADIUS)],
                ]
                activation_values = [
                    _dequantize_tgt(params, tuple(activation.shape))
                    for params in activation_params
                ]
                for weight_route in (0, 1):
                    for activation_route in (0, 1):
                        candidate_index = (
                            family * 4 + weight_route * 2 + activation_route
                        )
                        candidate_output = _linear_output(
                            activation_values[activation_route],
                            weight_values[weight_route],
                        )
                        losses[candidate_index].append(
                            _mse(
                                candidate_output,
                                reference_outputs[sample_index],
                            )
                        )

        selected = _select_candidate(losses)
        family = selected // 4
        family_candidate = selected % 4
        weight_route = family_candidate // 2
        activation_route = family_candidate % 2
        state = {
            "family": int(family),
            "mode": int(activation_route),
            "radius": int(
                _BASE_RADIUS if activation_route == 0 else _ENHANCED_RADIUS
            ),
            "importance": family_dynamic_importance[family]
            .detach()
            .contiguous(),
            "quadratic": (
                family_dynamic_quadratic[family].detach().contiguous()
                if family_dynamic_quadratic[family] is not None
                else None
            ),
            "permutation": permutation,
            "smooth_balance": smooth_balance,
        }
        return {
            "weight_params": family_weight_params[family][weight_route],
            "activation_state": state,
        }


def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    if type(activation_state) is not dict:
        raise ValueError("activation_state must be a dict")
    family = int(activation_state["family"])
    radius = int(activation_state["radius"])
    importance = activation_state["importance"]
    quadratic = activation_state["quadratic"]
    permutation = activation_state["permutation"]
    smooth_balance = activation_state["smooth_balance"]
    mode = int(activation_state["mode"])
    with torch.no_grad():
        activation = _apply_linear_family(
            _dequantize_src(activation_quant, activation_scale),
            family,
            permutation,
            smooth_balance,
        )
        return _quantize_hierarchical(
            activation,
            radius,
            importance if mode == 1 else None,
            quadratic if mode == 1 else None,
            refine_quadratic=(mode == 1),
        )


def _attention_importance(
    samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    repeat = q_num_heads // kv_num_heads
    q_energy = torch.zeros((q_num_heads, head_dim), dtype=torch.float32)
    k_energy = torch.zeros((kv_num_heads, head_dim), dtype=torch.float32)
    q_count = 0
    k_count = 0
    balance_patterns = []
    for q, k, _ in samples:
        q_heads = q.reshape(q.shape[0], q_num_heads, head_dim)
        k_heads = k.reshape(k.shape[0], kv_num_heads, head_dim)
        sample_q_energy = torch.mean(q_heads.square(), dim=0).reshape(
            kv_num_heads, repeat, head_dim
        ).mean(dim=1)
        sample_k_energy = torch.mean(k_heads.square(), dim=0)
        sample_pattern = _ATTENTION_BALANCE_EXPONENT * (
            torch.log2(sample_k_energy.clamp_min(1.0e-24))
            - torch.log2(sample_q_energy.clamp_min(1.0e-24))
        )
        sample_blocks = sample_pattern.reshape(kv_num_heads, -1, 64)
        sample_blocks = sample_blocks - torch.mean(
            sample_blocks, dim=-1, keepdim=True
        )
        balance_patterns.append(sample_blocks.flatten())
        q_energy += torch.sum(q_heads.square(), dim=0)
        k_energy += torch.sum(k_heads.square(), dim=0)
        q_count += int(q.shape[0])
        k_count += int(k.shape[0])
    q_energy = q_energy / max(q_count, 1)
    k_energy = k_energy / max(k_count, 1)
    q_importance = k_energy.repeat_interleave(repeat, dim=0)
    served_q_energy = q_energy.reshape(kv_num_heads, repeat, head_dim).mean(dim=1)
    k_importance = served_q_energy
    permutation = torch.argsort(
        served_q_energy + k_energy, dim=-1, stable=True
    ).to(torch.int64).contiguous()
    log_balance = _ATTENTION_BALANCE_EXPONENT * (
        torch.log2(k_energy.clamp_min(1.0e-24))
        - torch.log2(served_q_energy.clamp_min(1.0e-24))
    )
    balance_blocks = log_balance.reshape(kv_num_heads, -1, 64)
    balance_blocks = balance_blocks - torch.mean(
        balance_blocks, dim=-1, keepdim=True
    )
    balance = torch.pow(
        2.0,
        torch.clamp(
            balance_blocks,
            min=-_ATTENTION_BALANCE_LOG2_LIMIT,
            max=_ATTENTION_BALANCE_LOG2_LIMIT,
        ),
    ).reshape(kv_num_heads, head_dim).contiguous()
    pattern_matrix = torch.stack(balance_patterns)
    normalized_patterns = pattern_matrix / torch.linalg.vector_norm(
        pattern_matrix, dim=1, keepdim=True
    ).clamp_min(1.0e-12)
    pattern_count = len(balance_patterns)
    if pattern_count < 2:
        balance_stability = -1.0
    else:
        similarities = normalized_patterns @ normalized_patterns.transpose(0, 1)
        balance_stability = float(
            (
                (torch.sum(similarities) - pattern_count)
                / (pattern_count * (pattern_count - 1))
            ).item()
        )
    return (
        _normalize_importance(q_importance.flatten()),
        _normalize_importance(k_importance.flatten()),
        permutation,
        balance,
        balance_stability,
    )


def _apply_attention_permutation(
    value: torch.Tensor,
    num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    permutation: torch.Tensor,
) -> torch.Tensor:
    if num_heads % kv_num_heads != 0:
        raise ValueError("attention heads must be divisible by KV heads")
    if tuple(permutation.shape) != (kv_num_heads, head_dim):
        raise ValueError("attention permutation has an invalid shape")
    heads = value.reshape(value.shape[0], num_heads, head_dim)
    head_permutation = permutation.repeat_interleave(
        num_heads // kv_num_heads, dim=0
    )
    indices = head_permutation.unsqueeze(0).expand(heads.shape[0], -1, -1)
    return torch.gather(heads, -1, indices).reshape(value.shape).contiguous()


def _apply_attention_balance(
    value: torch.Tensor,
    num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    balance: torch.Tensor,
    *,
    inverse: bool,
) -> torch.Tensor:
    if num_heads % kv_num_heads != 0:
        raise ValueError("attention heads must be divisible by KV heads")
    if tuple(balance.shape) != (kv_num_heads, head_dim):
        raise ValueError("attention balance has an invalid shape")
    head_balance = balance.repeat_interleave(
        num_heads // kv_num_heads, dim=0
    )
    if inverse:
        head_balance = head_balance.reciprocal()
    heads = value.reshape(value.shape[0], num_heads, head_dim)
    return (heads * head_balance.unsqueeze(0)).reshape(value.shape).contiguous()


def _attention_scale_dispersion(calib_qkv_list: list) -> float:
    deviations = []
    for sample in calib_qkv_list:
        for stream in ("q", "k"):
            scale = sample[stream][1].detach().to(
                device="cpu", dtype=torch.float32
            )
            log_scale = torch.log2(scale.clamp_min(_MIN_SCALE_FACTOR))
            deviations.append(
                float(torch.std(log_scale, correction=0).item())
            )
    return sum(deviations) / len(deviations)


def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    if type(calib_qkv_list) is not list or not calib_qkv_list:
        raise ValueError("calib_qkv_list must be a non-empty list")
    for number in (q_num_heads, kv_num_heads, head_dim):
        if type(number) is not int or number <= 0:
            raise ValueError("head counts and head_dim must be positive ints")
    if q_num_heads % kv_num_heads != 0:
        raise ValueError("q_num_heads must be divisible by kv_num_heads")

    with torch.no_grad():
        scale_dispersion = _attention_scale_dispersion(calib_qkv_list)
        samples = []
        for sample in calib_qkv_list:
            q = _dequantize_src(*sample["q"])
            k = _dequantize_src(*sample["k"])
            v = _dequantize_src(*sample["v"])
            samples.append((q, k, v))
        (
            q_importance,
            k_importance,
            permutation,
            balance,
            balance_stability,
        ) = _attention_importance(samples, q_num_heads, kv_num_heads, head_dim)

        # bundle: (q route, k route, v route); route 0=baseline,
        # 1=expanded unweighted, 2=expanded sensitivity-weighted.
        bundles = [(0, 0, 0), (1, 1, 1), (2, 2, 1), (2, 2, 0)]
        losses = [[] for _ in bundles]
        hadamard_losses = []
        permuted_hadamard_losses = []
        balanced_losses = []
        balanced_hadamard_losses = []
        enable_hadamard = (
            head_dim % 64 == 0
            and scale_dispersion <= _ATTENTION_TRANSFORM_MAX_LOG_SCALE_STD
        )
        enable_balance = (
            head_dim % 64 == 0
            and balance_stability
            >= _stability_threshold(kv_num_heads * head_dim)
        )
        for q_ref, k_ref, v_ref in samples:
            reference_output = _attention_output(
                q_ref, k_ref, v_ref, q_num_heads, kv_num_heads, head_dim
            )
            q_route_map = _quantize_hierarchical_multi(
                q_ref,
                (_BASE_RADIUS, _ENHANCED_RADIUS),
                weighted_radius=_ENHANCED_RADIUS,
                importance=q_importance,
            )
            k_route_map = _quantize_hierarchical_multi(
                k_ref,
                (_BASE_RADIUS, _ENHANCED_RADIUS),
                weighted_radius=_ENHANCED_RADIUS,
                importance=k_importance,
            )
            v_route_map = _quantize_hierarchical_multi(
                v_ref, (_BASE_RADIUS, _ENHANCED_RADIUS)
            )
            q_routes = [
                q_route_map[("unweighted", _BASE_RADIUS)],
                q_route_map[("unweighted", _ENHANCED_RADIUS)],
                q_route_map[("weighted", _ENHANCED_RADIUS)],
            ]
            k_routes = [
                k_route_map[("unweighted", _BASE_RADIUS)],
                k_route_map[("unweighted", _ENHANCED_RADIUS)],
                k_route_map[("weighted", _ENHANCED_RADIUS)],
            ]
            v_routes = [
                v_route_map[("unweighted", _BASE_RADIUS)],
                v_route_map[("unweighted", _ENHANCED_RADIUS)],
            ]
            q_values = [_dequantize_tgt(p, tuple(q_ref.shape)) for p in q_routes]
            k_values = [_dequantize_tgt(p, tuple(k_ref.shape)) for p in k_routes]
            v_values = [_dequantize_tgt(p, tuple(v_ref.shape)) for p in v_routes]
            probability_cache = {}
            value_head_cache = {}
            for q_route, k_route, v_route in bundles:
                pair = (q_route, k_route)
                if pair not in probability_cache:
                    probability_cache[pair] = _attention_probabilities(
                        q_values[q_route],
                        k_values[k_route],
                        q_num_heads,
                        kv_num_heads,
                        head_dim,
                    )
                if v_route not in value_head_cache:
                    value_head_cache[v_route] = _attention_value_heads(
                        v_values[v_route],
                        q_num_heads,
                        kv_num_heads,
                        head_dim,
                    )
            for index, (q_route, k_route, v_route) in enumerate(bundles):
                candidate_output = _attention_from_probabilities(
                    probability_cache[(q_route, k_route)],
                    value_head_cache[v_route],
                    int(q_ref.shape[0]),
                    q_num_heads,
                    head_dim,
                )
                losses[index].append(_mse(candidate_output, reference_output))


            if enable_balance:
                balanced_q = _apply_attention_balance(
                    q_ref,
                    q_num_heads,
                    kv_num_heads,
                    head_dim,
                    balance,
                    inverse=False,
                )
                balanced_k = _apply_attention_balance(
                    k_ref,
                    kv_num_heads,
                    kv_num_heads,
                    head_dim,
                    balance,
                    inverse=True,
                )
                balanced_q_value = _dequantize_tgt(
                    _quantize_hierarchical(balanced_q, _ENHANCED_RADIUS),
                    tuple(q_ref.shape),
                )
                balanced_k_value = _dequantize_tgt(
                    _quantize_hierarchical(balanced_k, _ENHANCED_RADIUS),
                    tuple(k_ref.shape),
                )
                balanced_output = _attention_output(
                    balanced_q_value,
                    balanced_k_value,
                    v_values[1],
                    q_num_heads,
                    kv_num_heads,
                    head_dim,
                )
                balanced_losses.append(_mse(balanced_output, reference_output))

            if enable_hadamard:
                transformed_q = _fwht64(q_ref)
                transformed_k = _fwht64(k_ref)
                transformed_q_params = _quantize_hierarchical(
                    transformed_q, _ENHANCED_RADIUS
                )
                transformed_k_params = _quantize_hierarchical(
                    transformed_k, _ENHANCED_RADIUS
                )
                transformed_q_value = _dequantize_tgt(
                    transformed_q_params, tuple(q_ref.shape)
                )
                transformed_k_value = _dequantize_tgt(
                    transformed_k_params, tuple(k_ref.shape)
                )
                transformed_output = _attention_output(
                    transformed_q_value,
                    transformed_k_value,
                    v_values[1],
                    q_num_heads,
                    kv_num_heads,
                    head_dim,
                )
                hadamard_losses.append(
                    _mse(transformed_output, reference_output)
                )
                permuted_q = _fwht64(
                    _apply_attention_permutation(
                        q_ref,
                        q_num_heads,
                        kv_num_heads,
                        head_dim,
                        permutation,
                    )
                )
                permuted_k = _fwht64(
                    _apply_attention_permutation(
                        k_ref,
                        kv_num_heads,
                        kv_num_heads,
                        head_dim,
                        permutation,
                    )
                )
                permuted_q_value = _dequantize_tgt(
                    _quantize_hierarchical(permuted_q, _ENHANCED_RADIUS),
                    tuple(q_ref.shape),
                )
                permuted_k_value = _dequantize_tgt(
                    _quantize_hierarchical(permuted_k, _ENHANCED_RADIUS),
                    tuple(k_ref.shape),
                )
                permuted_output = _attention_output(
                    permuted_q_value,
                    permuted_k_value,
                    v_values[1],
                    q_num_heads,
                    kv_num_heads,
                    head_dim,
                )
                permuted_hadamard_losses.append(
                    _mse(permuted_output, reference_output)
                )
                if enable_balance:
                    balanced_h_q_value = _dequantize_tgt(
                        _quantize_hierarchical(
                            _fwht64(balanced_q), _ENHANCED_RADIUS
                        ),
                        tuple(q_ref.shape),
                    )
                    balanced_h_k_value = _dequantize_tgt(
                        _quantize_hierarchical(
                            _fwht64(balanced_k), _ENHANCED_RADIUS
                        ),
                        tuple(k_ref.shape),
                    )
                    balanced_h_output = _attention_output(
                        balanced_h_q_value,
                        balanced_h_k_value,
                        v_values[1],
                        q_num_heads,
                        kv_num_heads,
                        head_dim,
                    )
                    balanced_hadamard_losses.append(
                        _mse(balanced_h_output, reference_output)
                    )

        selected = _select_attention_candidate(losses)
        q_route, k_route, v_route = bundles[selected]
        transform = 0
        selected_losses = losses[selected]
        if enable_hadamard:
            transform_gains = _relative_gains(
                losses[selected], hadamard_losses
            )
            if (
                sum(transform_gains) / len(transform_gains)
                >= _ATTENTION_TRANSFORM_MEAN_GAIN
                and min(transform_gains) >= _ATTENTION_TRANSFORM_WORST_GAIN
            ):
                q_route, k_route, v_route = (1, 1, 1)
                transform = 1
                selected_losses = hadamard_losses
            permuted_transform_gains = _relative_gains(
                selected_losses, permuted_hadamard_losses
            )
            if (
                sum(permuted_transform_gains) / len(permuted_transform_gains)
                >= _ATTENTION_PERMUTED_TRANSFORM_MEAN_GAIN
                and min(permuted_transform_gains)
                >= _ATTENTION_PERMUTED_TRANSFORM_WORST_GAIN
            ):
                q_route, k_route, v_route = (1, 1, 1)
                transform = 2
                selected_losses = permuted_hadamard_losses
        balanced_candidates = []
        if enable_balance:
            balanced_candidates.append((3, balanced_losses))
            if enable_hadamard:
                balanced_candidates.append((4, balanced_hadamard_losses))
        best_balanced_gain = 0.0
        for candidate_transform, candidate_losses in balanced_candidates:
            gains = _relative_gains(selected_losses, candidate_losses)
            mean_gain = sum(gains) / len(gains)
            if (
                mean_gain >= _ATTENTION_BALANCED_TRANSFORM_MEAN_GAIN
                and min(gains) >= _ATTENTION_BALANCED_TRANSFORM_WORST_GAIN
                and mean_gain > best_balanced_gain
            ):
                q_route, k_route, v_route = (1, 1, 1)
                transform = candidate_transform
                best_balanced_gain = mean_gain
        q_state = {
            "route": int(q_route),
            "radius": int(_BASE_RADIUS if q_route == 0 else _ENHANCED_RADIUS),
            "importance": q_importance,
            "transform": int(transform),
            "kv_num_heads": int(kv_num_heads),
            "permutation": permutation,
            "balance": balance,
        }
        k_state = {
            "route": int(k_route),
            "radius": int(_BASE_RADIUS if k_route == 0 else _ENHANCED_RADIUS),
            "importance": k_importance,
            "transform": int(transform),
            "kv_num_heads": int(kv_num_heads),
            "permutation": permutation,
            "balance": balance,
        }
        v_state = {
            "route": int(v_route),
            "radius": int(_BASE_RADIUS if v_route == 0 else _ENHANCED_RADIUS),
            "transform": 0,
        }
        return {"q_state": q_state, "k_state": k_state, "v_state": v_state}


def _dynamic_attention(
    quant: torch.Tensor,
    scale: torch.Tensor,
    num_heads: int,
    head_dim: int,
    state: Any,
    stream: str,
) -> dict[str, torch.Tensor]:
    if type(num_heads) is not int or num_heads <= 0:
        raise ValueError("num_heads must be a positive int")
    if type(head_dim) is not int or head_dim <= 0:
        raise ValueError("head_dim must be a positive int")
    if quant.ndim != 2 or quant.shape[-1] != num_heads * head_dim:
        raise ValueError(f"{stream} width does not match num_heads * head_dim")
    if type(state) is not dict:
        raise ValueError(f"{stream}_state must be a dict")
    route = int(state["route"])
    radius = int(state["radius"])
    transform = int(state.get("transform", 0))
    importance = state.get("importance") if route == 2 else None
    with torch.no_grad():
        value = _dequantize_src(quant, scale)
        if transform in (1, 2, 3, 4):
            if stream not in {"q", "k"} or head_dim % 64 != 0:
                raise ValueError(
                    "attention transform requires a Q/K head_dim divisible by 64"
                )
            if transform == 2:
                value = _apply_attention_permutation(
                    value,
                    num_heads,
                    int(state["kv_num_heads"]),
                    head_dim,
                    state["permutation"],
                )
            elif transform in (3, 4):
                value = _apply_attention_balance(
                    value,
                    num_heads,
                    int(state["kv_num_heads"]),
                    head_dim,
                    state["balance"],
                    inverse=stream == "k",
                )
            if transform in (1, 2, 4):
                value = _fwht64(value)
        elif transform != 0:
            raise ValueError("unknown attention transform")
        return _quantize_hierarchical(value, radius, importance)


def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    return _dynamic_attention(
        q_quant, q_scale, q_num_heads, head_dim, q_state, "q"
    )


def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    return _dynamic_attention(
        k_quant, k_scale, kv_num_heads, head_dim, k_state, "k"
    )


def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    return _dynamic_attention(
        v_quant, v_scale, kv_num_heads, head_dim, v_state, "v"
    )
