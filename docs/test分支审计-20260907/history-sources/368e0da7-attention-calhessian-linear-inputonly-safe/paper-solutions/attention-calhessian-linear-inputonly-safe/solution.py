from __future__ import annotations

import math
from typing import Any

import torch


_MIN_SCALE_FACTOR = 2.0**-48
_MAX_SCALE_FACTOR = 49152.0
_MANTISSA_MAX = 1.75
_MANTISSA_STEP = 0.25
_BASE_RADIUS = 3
_ENHANCED_RADIUS = 8
_ATTENTION_RADIUS = 3
_ATTENTION_QK_RADIUS = 3
_ATTENTION_SCALE_OFFSETS = (-2, -1, 0, 1, 2, 3)
_ATTENTION_V_SCALE_OFFSETS = (-1, 0, 1, 2, 3)
_SWITCH_MEAN_GAIN = 0.001
_SWITCH_WORST_GAIN = -0.1
_IMPORTANCE_MIN = 0.25
_IMPORTANCE_MAX = 4.0
_ATTENTION_IMPORTANCE_MIN = 0.1
_ATTENTION_IMPORTANCE_MAX = 8.0
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
_LINEAR_HADAMARD_SIZE = 64
_LINEAR_SMOOTH_LOG2_LIMIT = 3.0
_LINEAR_SMOOTH_CENTER_WIDTH = 64
_STABILITY_STANDARD_DEVIATIONS = 3.0
_PSOT_MAX_RANK = 1
_FIT_MEAN_GAIN = 0.01


def _dequantize_src(quant: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    blocked = quant.detach().to(device="cpu").unflatten(-1, (-1, 16))
    value = blocked * scale.detach().to(device="cpu").unsqueeze(-1)
    return value.flatten(-2, -1).to(torch.bfloat16).to(torch.float32).contiguous()


def _fwht(value: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    if value.ndim < 1 or value.shape[-1] % block_size != 0:
        raise ValueError(f"Hadamard transform requires a width divisible by {block_size}")
    original_shape = tuple(value.shape)
    transformed = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    transformed = transformed.reshape(original_shape[:-1] + (-1, block_size))
    log2_size = int(math.log2(block_size))
    for i in range(log2_size):
        half_width = 1 << i
        groups = transformed.reshape(
            transformed.shape[:-1] + (block_size // (2 * half_width), 2, half_width)
        )
        left = groups[..., 0, :]
        right = groups[..., 1, :]
        transformed = torch.stack((left + right, left - right), dim=-2).reshape(
            transformed.shape
        )
    return (transformed * (1.0 / math.sqrt(block_size))).reshape(original_shape).contiguous()


def _apply_linear_family(
    value: torch.Tensor,
    family: int,
    permutation: torch.Tensor,
    smooth_balance: torch.Tensor,
    *,
    inverse_smooth: bool = False,
    hadamard_size: int = 64,
) -> torch.Tensor:
    if family == 0:
        return value
    if family == 1:
        return _fwht(value, hadamard_size)
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
        return _fwht(value * factor, hadamard_size)
    if family == 5:
        factor = (
            smooth_balance.reciprocal()
            if inverse_smooth
            else smooth_balance
        )
        return _fwht(value, hadamard_size) * factor
    raise ValueError("unknown Linear transform family")


def _linear_smooth_balance(
    weight: torch.Tensor,
    activation_energy: torch.Tensor,
    activation_peak: torch.Tensor,
    activation_count: int,
    alpha: float = _LINEAR_SMOOTH_ALPHA,
) -> torch.Tensor:
    weight_rms = torch.mean(weight.square(), dim=0).clamp_min(1.0e-24).sqrt()
    activation_rms = (
        activation_energy / max(activation_count, 1)
    ).clamp_min(1.0e-24).sqrt()
    log_balance = (
        (1.0 - alpha) * torch.log2(weight_rms)
        - alpha * torch.log2(activation_rms)
    )
    center_width = math.gcd(int(log_balance.numel()), _LINEAR_SMOOTH_CENTER_WIDTH)
    log_balance = log_balance.reshape(-1, center_width)
    log_balance = log_balance - torch.mean(
        log_balance, dim=-1, keepdim=True
    )
    balance = torch.pow(
        2.0,
        torch.clamp(
            log_balance,
            min=-_LINEAR_SMOOTH_LOG2_LIMIT,
            max=_LINEAR_SMOOTH_LOG2_LIMIT,
        ),
    ).flatten()
    log_peak = torch.log2(activation_peak.clamp_min(1.0e-24))
    peak_center = torch.median(log_peak)
    peak_mad = torch.median(torch.abs(log_peak - peak_center)).clamp_min(0.25)
    outlier_channels = log_peak > peak_center + 2.0 * peak_mad
    return torch.where(outlier_channels, balance, 1.0).contiguous()


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


def _peak_suppression_reflectors(
    activations: list[torch.Tensor],
) -> torch.Tensor:
    channels = int(activations[0].shape[-1])
    block_count = channels // 64
    rows = torch.cat(
        [value.reshape(-1, block_count, 64) for value in activations], dim=0
    )
    covariance = torch.einsum("nbi,nbj->bij", rows, rows) / max(
        int(rows.shape[0]), 1
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    base_rms = torch.mean(rows.square(), dim=(0, 2)).clamp_min(1.0e-24).sqrt()
    noise_edge = 1.25 * (
        1.0 + math.sqrt(64.0 / max(int(rows.shape[0]), 1))
    )
    indices = torch.arange(64, dtype=torch.int64)
    reflectors = []
    for rank_index in range(_PSOT_MAX_RANK):
        direction = eigenvectors[..., -(rank_index + 1)]
        for previous in reflectors:
            projection = torch.sum(direction * previous, dim=-1, keepdim=True)
            direction = direction - 2.0 * projection * previous
        target = torch.where(
            torch.remainder(torch.div(indices, 16, rounding_mode="floor"), 2)
            == 0,
            0.125,
            -0.125,
        )
        align_sign = torch.where(
            torch.sum(direction * target, dim=-1, keepdim=True) < 0.0,
            -1.0,
            1.0,
        )
        direction = direction * align_sign
        reflector = direction - target
        reflector = reflector / torch.linalg.vector_norm(
            reflector, dim=-1, keepdim=True
        ).clamp_min(1.0e-12)
        direction_rms = eigenvalues[..., -(rank_index + 1)].clamp_min(0.0).sqrt()
        active = direction_rms / base_rms > noise_edge
        reflector = torch.where(active.unsqueeze(-1), reflector, 0.0)
        reflectors.append(reflector.contiguous())
    return torch.stack(reflectors, dim=0).contiguous()


def _apply_reflectors(
    value: torch.Tensor,
    reflectors: torch.Tensor,
) -> torch.Tensor:
    original_shape = tuple(value.shape)
    blocks = value.reshape(original_shape[:-1] + (-1, 64))
    for reflector in reflectors:
        projection = torch.einsum("...bi,bi->...b", blocks, reflector)
        blocks = blocks - 2.0 * projection.unsqueeze(-1) * reflector
    return blocks.reshape(original_shape).contiguous()


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
    linear: torch.Tensor | None = None,
) -> torch.Tensor:
    vectors = error.flatten(-3, -1)
    block_error = torch.einsum(
        "...bi,bij,...bj->...b", vectors, quadratic, vectors
    )
    if linear is not None:
        block_error = block_error + 2.0 * torch.sum(
            vectors * linear.flatten(-3, -1), dim=-1
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


def _take_best_vectorized_candidate(
    error: torch.Tensor,
    scale_factor: torch.Tensor,
    lv2: torch.Tensor,
    lv3: torch.Tensor,
    mantissa: torch.Tensor,
    candidate_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_dim = error.ndim - 5
    selected_error = error.index_select(candidate_dim, candidate_indices)
    best_index = torch.argmin(selected_error, dim=candidate_dim, keepdim=True)

    def gather_best(value: torch.Tensor) -> torch.Tensor:
        selected = value.index_select(candidate_dim, candidate_indices)
        gather_shape = list(selected.shape)
        gather_shape[candidate_dim] = 1
        index = best_index.expand(gather_shape)
        return torch.gather(selected, candidate_dim, index).squeeze(candidate_dim)

    return (
        gather_best(error),
        gather_best(scale_factor),
        gather_best(lv2),
        gather_best(lv3),
        gather_best(mantissa),
    )


def _quantize_hierarchical_multi(
    value: torch.Tensor,
    unweighted_radii: tuple[int, ...],
    weighted_radius: int | None = None,
    importance: torch.Tensor | None = None,
    quadratic: torch.Tensor | None = None,
    linear: torch.Tensor | None = None,
    candidate_offsets: tuple[int, ...] | None = None,
) -> dict[tuple[str, int], dict[str, torch.Tensor]]:
    if not isinstance(value, torch.Tensor) or value.ndim < 1:
        raise ValueError("value must be a non-scalar tensor")
    if not unweighted_radii and weighted_radius is None:
        raise ValueError("at least one quantization route is required")
    if weighted_radius is not None and importance is None:
        raise ValueError("weighted quantization requires importance")
    if quadratic is not None and weighted_radius is None:
        raise ValueError("quadratic requires weighted quantization")
    if linear is not None and weighted_radius is None:
        raise ValueError("linear term requires weighted quantization")
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
    block_linear = None
    if linear is not None:
        linear_value = linear.detach().to(device="cpu", dtype=torch.float32)
        try:
            linear_value = torch.broadcast_to(linear_value, original_shape)
        except RuntimeError as error:
            raise ValueError("linear term cannot be broadcast to the input shape") from error
        if not bool(torch.isfinite(linear_value).all().item()):
            raise ValueError("linear term must contain finite values")
        block_linear = linear_value.contiguous().reshape(blocked_shape)
    absolute = torch.abs(x)
    input_sign = torch.sign(x)
    block_peak = torch.amax(absolute, dim=(-3, -2, -1), keepdim=True)
    maximum_radius = max(requested_radii)
    if candidate_offsets is None:
        active_offsets = tuple(range(-maximum_radius, maximum_radius + 1))
        candidates = _e6m2_candidates(block_peak, maximum_radius)
    else:
        active_offsets = candidate_offsets
        offset_radius = max(abs(offset) for offset in active_offsets)
        all_candidates = _e6m2_candidates(block_peak, offset_radius)
        indices = torch.tensor(
            [offset + offset_radius for offset in active_offsets],
            dtype=torch.int64,
        )
        candidates = all_candidates.index_select(-1, indices)
    unweighted_best = {radius: None for radius in unweighted_radii}
    weighted_best = None

    # Calibration route searches have no cross-channel quadratic. Evaluate all
    # scale offsets in one tensor so the small reductions do not pay Python and
    # dispatcher overhead once per offset. Hessian-guided dynamic paths retain
    # the scalar loop below to keep their memory footprint bounded.
    if block_quadratic is None and block_linear is None:
        candidate_dim = x.ndim - 4
        candidate_x = x.unsqueeze(candidate_dim)
        candidate_absolute = absolute.unsqueeze(candidate_dim)
        candidate_sign = input_sign.unsqueeze(candidate_dim)
        candidate_scales = candidates.movedim(-1, candidate_dim)
        candidate_weight = (
            None if weight is None else weight.unsqueeze(candidate_dim)
        )
        mantissas = []
        squared_errors = []
        for multiplier in (1.0, 2.0, 4.0):
            mantissa, squared_error = _quantize_multiplier(
                candidate_x,
                candidate_absolute,
                candidate_sign,
                candidate_scales,
                multiplier,
            )
            mantissas.append(mantissa)
            squared_errors.append(squared_error)
        mantissa_tuple = tuple(mantissas)

        if unweighted_radii:
            unweighted_errors = tuple(
                _reduce_quartet_error(error, None)
                for error in squared_errors
            )
            error, lv2, lv3, mantissa = _select_at_scale(
                mantissa_tuple, unweighted_errors
            )
            for radius in unweighted_radii:
                active_indices = torch.tensor(
                    [
                        index
                        for index, offset in enumerate(active_offsets)
                        if radius == maximum_radius or abs(offset) <= radius
                    ],
                    dtype=torch.int64,
                )
                unweighted_best[radius] = _take_best_vectorized_candidate(
                    error,
                    candidate_scales,
                    lv2,
                    lv3,
                    mantissa,
                    active_indices,
                )

        if weighted_radius is not None:
            weighted_errors = tuple(
                _reduce_quartet_error(error, candidate_weight)
                for error in squared_errors
            )
            error, lv2, lv3, mantissa = _select_at_scale(
                mantissa_tuple, weighted_errors
            )
            active_indices = torch.tensor(
                [
                    index
                    for index, offset in enumerate(active_offsets)
                    if weighted_radius == maximum_radius
                    or abs(offset) <= weighted_radius
                ],
                dtype=torch.int64,
            )
            weighted_best = _take_best_vectorized_candidate(
                error,
                candidate_scales,
                lv2,
                lv3,
                mantissa,
                active_indices,
            )

        results = {
            ("unweighted", radius): _finalize_candidate(best, input_sign)
            for radius, best in unweighted_best.items()
        }
        if weighted_radius is not None:
            results[("weighted", weighted_radius)] = _finalize_candidate(
                weighted_best, input_sign
            )
        return results

    for candidate_index in range(candidates.shape[-1]):
        offset = active_offsets[candidate_index]
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
        reconstruction_errors = (
            input_sign * mantissa_1 * scale_factor - x,
            input_sign * mantissa_2 * scale_factor * 2.0 - x,
            input_sign * mantissa_4 * scale_factor * 4.0 - x,
        )

        active_unweighted = [
            radius
            for radius in unweighted_radii
            if radius == maximum_radius or abs(offset) <= radius
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

        if weighted_radius is not None and (
            weighted_radius == maximum_radius or abs(offset) <= weighted_radius
        ):
            weighted_errors = (
                _reduce_quartet_error(squared_error_1, weight),
                _reduce_quartet_error(squared_error_2, weight),
                _reduce_quartet_error(squared_error_4, weight),
            )
            if block_linear is not None:
                weighted_errors = tuple(
                    weighted_error
                    + _reduce_quartet_error(
                        2.0 * block_linear * reconstruction_error, None
                    )
                    for weighted_error, reconstruction_error in zip(
                        weighted_errors, reconstruction_errors
                    )
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
                    reconstructed - x, block_quadratic, block_linear
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
        results[("weighted", weighted_radius)] = _finalize_candidate(
            weighted_best, input_sign
        )
    return results


def _quantize_hierarchical(
    value: torch.Tensor,
    radius: int,
    importance: torch.Tensor | None = None,
    quadratic: torch.Tensor | None = None,
    linear: torch.Tensor | None = None,
    candidate_offsets: tuple[int, ...] | None = None,
) -> dict[str, torch.Tensor]:
    if importance is None and linear is None:
        return _quantize_hierarchical_multi(
            value, (radius,), candidate_offsets=candidate_offsets
        )[
            ("unweighted", radius)
        ]
    return _quantize_hierarchical_multi(
        value,
        (),
        weighted_radius=radius,
        importance=importance,
        quadratic=quadratic,
        linear=linear,
        candidate_offsets=candidate_offsets,
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


def _normalize_linear_importance(importance: torch.Tensor) -> torch.Tensor:
    flat = importance.detach().to(device="cpu", dtype=torch.float32).flatten()
    if flat.numel() % 64 != 0:
        raise ValueError("importance width must be divisible by 64")
    blocks = flat.reshape(-1, 64)
    means = torch.mean(blocks, dim=-1, keepdim=True).clamp_min(1.0e-12)
    normalized = blocks / means
    return torch.clamp(normalized, min=0.5, max=2.0).flatten().contiguous()


def _normalize_importance_wide(importance: torch.Tensor) -> torch.Tensor:
    """Normalize importance with wider bounds for attention Q/K."""
    flat = importance.detach().to(device="cpu", dtype=torch.float32).flatten()
    if flat.numel() % 64 != 0:
        raise ValueError("importance width must be divisible by 64")
    blocks = flat.reshape(-1, 64)
    means = torch.mean(blocks, dim=-1, keepdim=True).clamp_min(1.0e-12)
    normalized = blocks / means
    return torch.clamp(
        normalized, min=_ATTENTION_IMPORTANCE_MIN, max=_ATTENTION_IMPORTANCE_MAX
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


def _probe_linear_losses(
    weight: torch.Tensor,
    activations: list[torch.Tensor],
    references: list[torch.Tensor],
) -> list[float]:
    weight_params = _quantize_hierarchical(weight, _BASE_RADIUS)
    weight_value = _dequantize_tgt(weight_params, tuple(weight.shape))
    losses = []
    for activation, reference in zip(activations, references):
        activation_params = _quantize_hierarchical(activation, _BASE_RADIUS)
        activation_value = _dequantize_tgt(
            activation_params, tuple(activation.shape)
        )
        losses.append(
            _mse(_linear_output(activation_value, weight_value), reference)
        )
    return losses


def _block_second_moment(
    values: list[torch.Tensor],
    *,
    require_independent_samples: bool = False,
    exact: bool = False,
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
    if exact:
        return moment.contiguous()
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


def _refine_weight_block_hessian(
    params: dict[str, torch.Tensor],
    target: torch.Tensor,
    hessian: torch.Tensor | None,
    rounds: int = 4,
) -> dict[str, torch.Tensor]:
    """Lower the calibration weight Hessian in legal mantissa steps."""
    if hessian is None:
        return params
    output_channels, channels = target.shape
    block_count = channels // 64
    if tuple(hessian.shape) != (block_count, 64, 64):
        return params
    refined = {key: value.clone() for key, value in params.items()}
    reconstructed = _dequantize_tgt(refined, tuple(target.shape))
    error = (reconstructed - target).reshape(output_channels, block_count, 64)
    mantissa = refined["mant"].reshape(output_channels, block_count, 64)
    sign = refined["sign"].reshape(output_channels, block_count, 64)
    effective_scale = (
        refined["scale_factor"]
        * refined["scale_lv2"]
        * refined["scale_lv3"]
    ).expand_as(refined["mant"]).reshape(output_channels, block_count, 64)
    target_sign = torch.sign(target).reshape(output_channels, block_count, 64)
    hessian_value = hessian.to(torch.float32)
    curvature = torch.diagonal(
        hessian_value, dim1=-2, dim2=-1
    ).unsqueeze(0)
    infinity = torch.tensor(float("inf"), dtype=torch.float32)
    for _ in range(rounds):
        gradient = torch.einsum("bij,obj->obi", hessian_value, error)
        step = _MANTISSA_STEP * effective_scale
        increment_sign = torch.where(sign != 0.0, sign, target_sign)
        increment = increment_sign * step
        decrement = -sign * step
        increment_change = torch.where(
            mantissa < _MANTISSA_MAX,
            2.0 * increment * gradient + increment.square() * curvature,
            infinity,
        )
        decrement_change = torch.where(
            mantissa > 0.0,
            2.0 * decrement * gradient + decrement.square() * curvature,
            infinity,
        )
        use_increment = increment_change <= decrement_change
        best_change = torch.where(
            use_increment, increment_change, decrement_change
        )
        best_value, best_index = torch.min(best_change, dim=-1)
        selected = torch.zeros_like(best_change, dtype=torch.bool)
        selected.scatter_(-1, best_index.unsqueeze(-1), True)
        selected &= best_value.unsqueeze(-1) < 0.0
        delta = torch.where(
            selected,
            torch.where(use_increment, increment, decrement),
            0.0,
        )
        candidate_mantissa = torch.where(
            use_increment,
            mantissa + _MANTISSA_STEP,
            mantissa - _MANTISSA_STEP,
        )
        mantissa.copy_(torch.where(selected, candidate_mantissa, mantissa))
        candidate_sign = torch.where(
            candidate_mantissa == 0.0,
            0.0,
            torch.where(
                use_increment & (sign == 0.0), increment_sign, sign
            ),
        )
        sign.copy_(torch.where(selected, candidate_sign, sign))
        error += delta
    return {key: value.contiguous() for key, value in refined.items()}


def _refine_weight_pair_hessian(
    params: dict[str, torch.Tensor],
    target: torch.Tensor,
    hessian: torch.Tensor | None,
    candidate_coordinates: int = 4,
    rounds: int = 1,
) -> dict[str, torch.Tensor]:
    """Escape single-coordinate minima with exact two-coordinate moves."""
    if hessian is None or candidate_coordinates < 2 or rounds <= 0:
        return params
    output_channels, channels = target.shape
    block_count = channels // 64
    if tuple(hessian.shape) != (block_count, 64, 64):
        return params
    refined = {key: value.clone() for key, value in params.items()}
    reconstructed = _dequantize_tgt(refined, tuple(target.shape))
    error = (reconstructed - target).reshape(output_channels, block_count, 64)
    mantissa = refined["mant"].reshape(output_channels, block_count, 64)
    sign = refined["sign"].reshape(output_channels, block_count, 64)
    effective_scale = (
        refined["scale_factor"]
        * refined["scale_lv2"]
        * refined["scale_lv3"]
    ).expand_as(refined["mant"]).reshape(output_channels, block_count, 64)
    target_sign = torch.sign(target).reshape(output_channels, block_count, 64)
    hessian_value = hessian.to(torch.float32)
    curvature = torch.diagonal(
        hessian_value, dim1=-2, dim2=-1
    ).unsqueeze(0)
    infinity = torch.tensor(float("inf"), dtype=torch.float32)
    coordinate_count = min(candidate_coordinates, 64)
    move_count = 2 * coordinate_count
    upper_triangle = torch.triu(
        torch.ones((move_count, move_count), dtype=torch.bool), diagonal=1
    )

    for _ in range(rounds):
        gradient = torch.einsum("bij,obj->obi", hessian_value, error)
        step = _MANTISSA_STEP * effective_scale
        increment_sign = torch.where(sign != 0.0, sign, target_sign)
        increments = torch.stack((increment_sign * step, -sign * step), dim=-1)
        valid_moves = torch.stack(
            (mantissa < _MANTISSA_MAX, mantissa > 0.0), dim=-1
        )
        single_change = (
            2.0 * increments * gradient.unsqueeze(-1)
            + increments.square() * curvature.unsqueeze(-1)
        )
        # Only test pairs whose constituent moves cannot improve the objective.
        eligible_change = torch.where(
            valid_moves & (single_change >= 0.0), single_change, infinity
        )
        coordinate_score = torch.min(eligible_change, dim=-1).values
        _, coordinate_index = torch.topk(
            coordinate_score,
            k=coordinate_count,
            dim=-1,
            largest=False,
        )
        gather_coordinate = coordinate_index.unsqueeze(-1).expand(
            output_channels, block_count, coordinate_count, 2
        )
        selected_change = torch.gather(
            eligible_change, 2, gather_coordinate
        ).reshape(output_channels, block_count, move_count)
        selected_delta = torch.gather(
            increments, 2, gather_coordinate
        ).reshape(output_channels, block_count, move_count)
        selected_index = coordinate_index.unsqueeze(-1).expand(
            output_channels, block_count, coordinate_count, 2
        ).reshape(output_channels, block_count, move_count)

        expanded_hessian = hessian_value.unsqueeze(0).expand(
            output_channels, -1, -1, -1
        )
        selected_rows = torch.gather(
            expanded_hessian,
            2,
            selected_index.unsqueeze(-1).expand(-1, -1, -1, 64),
        )
        selected_cross = torch.gather(
            selected_rows,
            3,
            selected_index.unsqueeze(-2).expand(-1, -1, move_count, -1),
        )
        pair_change = (
            selected_change.unsqueeze(-1)
            + selected_change.unsqueeze(-2)
            + 2.0
            * selected_delta.unsqueeze(-1)
            * selected_delta.unsqueeze(-2)
            * selected_cross
        )
        valid_pair = (
            upper_triangle.reshape(1, 1, move_count, move_count)
            & (selected_index.unsqueeze(-1) != selected_index.unsqueeze(-2))
            & torch.isfinite(pair_change)
        )
        pair_change = torch.where(valid_pair, pair_change, infinity)
        best_change, flat_pair_index = torch.min(
            pair_change.reshape(output_channels, block_count, -1), dim=-1
        )
        first_move = torch.div(
            flat_pair_index, move_count, rounding_mode="floor"
        )
        second_move = torch.remainder(flat_pair_index, move_count)
        accepted = best_change < 0.0
        first_coordinate = torch.gather(
            selected_index, -1, first_move.unsqueeze(-1)
        ).squeeze(-1)
        second_coordinate = torch.gather(
            selected_index, -1, second_move.unsqueeze(-1)
        ).squeeze(-1)
        first_delta = torch.gather(
            selected_delta, -1, first_move.unsqueeze(-1)
        ).squeeze(-1)
        second_delta = torch.gather(
            selected_delta, -1, second_move.unsqueeze(-1)
        ).squeeze(-1)
        pair_delta = torch.zeros_like(error)
        pair_delta.scatter_add_(
            -1,
            first_coordinate.unsqueeze(-1),
            torch.where(accepted, first_delta, 0.0).unsqueeze(-1),
        )
        pair_delta.scatter_add_(
            -1,
            second_coordinate.unsqueeze(-1),
            torch.where(accepted, second_delta, 0.0).unsqueeze(-1),
        )
        reconstructed_blocks = sign * mantissa * effective_scale + pair_delta
        mantissa.copy_(
            torch.clamp(
                torch.abs(reconstructed_blocks) / effective_scale,
                min=0.0,
                max=_MANTISSA_MAX,
            )
        )
        sign.copy_(
            torch.where(
                mantissa == 0.0, 0.0, torch.sign(reconstructed_blocks)
            )
        )
        error += pair_delta
    return {key: value.contiguous() for key, value in refined.items()}


def _refine_weight_lv3_hessian(
    params: dict[str, torch.Tensor],
    target: torch.Tensor,
    hessian: torch.Tensor | None,
    rounds: int = 2,
) -> dict[str, torch.Tensor]:
    """Flip one four-value HiF4 subtree per block under the exact Hessian."""
    if hessian is None or rounds <= 0:
        return params
    output_channels, channels = target.shape
    block_count = channels // 64
    if tuple(hessian.shape) != (block_count, 64, 64):
        return params
    refined = {key: value.clone() for key, value in params.items()}
    reconstructed = _dequantize_tgt(refined, tuple(target.shape))
    target_quartets = target.reshape(output_channels, block_count, 16, 4)
    error = (reconstructed - target).reshape_as(target_quartets)
    mantissa = refined["mant"].reshape_as(target_quartets)
    sign = refined["sign"].reshape_as(target_quartets)
    scale_factor = refined["scale_factor"].reshape(
        output_channels, block_count, 1, 1
    )
    lv2 = refined["scale_lv2"].expand(
        output_channels, block_count, 8, 2, 1
    ).reshape(output_channels, block_count, 16, 1)
    lv3 = refined["scale_lv3"].reshape(
        output_channels, block_count, 16, 1
    )
    hessian_value = hessian.to(torch.float32)
    quartet_index = torch.arange(64, dtype=torch.int64).reshape(16, 4)
    hessian_rows = hessian_value[:, quartet_index, :]
    hessian_quartets = torch.gather(
        hessian_rows,
        -1,
        quartet_index.reshape(1, 16, 1, 4).expand(block_count, -1, 4, -1),
    )

    for _ in range(rounds):
        flat_error = error.reshape(output_channels, block_count, 64)
        gradient = torch.einsum(
            "bij,obj->obi", hessian_value, flat_error
        ).reshape_as(error)
        candidate_lv3 = 3.0 - lv3
        effective_scale = scale_factor * lv2 * candidate_lv3
        candidate_mantissa = torch.clamp(
            torch.round(
                torch.abs(target_quartets)
                / effective_scale
                / _MANTISSA_STEP
            )
            * _MANTISSA_STEP,
            min=0.0,
            max=_MANTISSA_MAX,
        )
        candidate_value = (
            torch.sign(target_quartets)
            * candidate_mantissa
            * effective_scale
        )
        current_value = target_quartets + error
        delta = candidate_value - current_value
        objective_change = (
            2.0 * torch.sum(delta * gradient, dim=-1)
            + torch.einsum(
                "obqi,bqij,obqj->obq",
                delta,
                hessian_quartets,
                delta,
            )
        )
        best_change, best_quartet = torch.min(objective_change, dim=-1)
        selected = torch.zeros_like(objective_change, dtype=torch.bool)
        selected.scatter_(-1, best_quartet.unsqueeze(-1), True)
        selected &= best_change.unsqueeze(-1) < 0.0
        selected_values = selected.unsqueeze(-1)
        lv3.copy_(torch.where(selected_values, candidate_lv3, lv3))
        mantissa.copy_(
            torch.where(selected_values, candidate_mantissa, mantissa)
        )
        sign.copy_(
            torch.where(
                selected_values,
                torch.where(
                    candidate_mantissa == 0.0,
                    0.0,
                    torch.sign(target_quartets),
                ),
                sign,
            )
        )
        error += torch.where(selected_values, delta, 0.0)
    return {key: value.contiguous() for key, value in refined.items()}


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
        raw_activation_peak = torch.zeros_like(raw_activation_energy)
        raw_sample_count = 0
        for activation in activation_refs:
            raw_activation_energy += torch.sum(activation.square(), dim=0)
            raw_activation_peak = torch.maximum(
                raw_activation_peak, torch.amax(torch.abs(activation), dim=0)
            )
            raw_sample_count += int(activation.shape[0])
        enable_quadratic = len(activation_refs) >= 2
        # Search for the best smooth alpha (OmniQuant LET)
        reference_outputs = [
            _linear_output(activation, weight_ref)
            for activation in activation_refs
        ]
        best_alpha = 0.3
        smooth_balance = _linear_smooth_balance(
            weight_ref,
            raw_activation_energy,
            raw_activation_peak,
            raw_sample_count,
            alpha=best_alpha,
        )
        if enable_quadratic:
            reference_outputs = [
                _linear_output(activation, weight_ref)
                for activation in activation_refs
            ]
            activation_log_rms = torch.log2(
                (
                    raw_activation_energy / max(raw_sample_count, 1)
                ).clamp_min(1e-24).sqrt()
            )
            activation_log_rms_std = float(
                torch.std(activation_log_rms, correction=0).item()
            )
            alpha_candidates = (
                (0.16, 0.24)
                if activation_log_rms_std > 8.0
                else (0.35, 0.5)
            )
            alpha_data = []
            alpha_losses = []
            for alpha in alpha_candidates:
                candidate_balance = _linear_smooth_balance(
                    weight_ref,
                    raw_activation_energy,
                    raw_activation_peak,
                    raw_sample_count,
                    alpha=alpha,
                )
                candidate_weight = _fwht(
                    weight_ref / candidate_balance, _LINEAR_HADAMARD_SIZE
                )
                candidate_activations = [
                    _fwht(
                        activation * candidate_balance,
                        _LINEAR_HADAMARD_SIZE,
                    )
                    for activation in activation_refs
                ]
                candidate_reflectors = _peak_suppression_reflectors(
                    candidate_activations
                )
                candidate_weight = _apply_reflectors(
                    candidate_weight, candidate_reflectors
                )
                candidate_activations = [
                    _apply_reflectors(activation, candidate_reflectors)
                    for activation in candidate_activations
                ]
                alpha_data.append(
                    (
                        candidate_balance,
                        candidate_weight,
                        candidate_activations,
                        candidate_reflectors,
                    )
                )
                if len(alpha_candidates) > 1:
                    alpha_losses.append(
                        _probe_linear_losses(
                            candidate_weight,
                            candidate_activations,
                            reference_outputs,
                        )
                    )
            alpha_route = (
                0
                if len(alpha_data) == 1
                else _select_attention_candidate(alpha_losses)
            )
            (
                smooth_balance,
                smoothed_weight,
                smoothed_activations,
                reflectors,
            ) = alpha_data[alpha_route]
            # Skip PSOT — directly quantize smoothed weight with importance
            activation_energy = torch.zeros(
                smoothed_weight.shape[-1], dtype=torch.float32
            )
            sample_count = 0
            for activation in smoothed_activations:
                activation_energy += torch.sum(
                    activation.square(), dim=0
                )
                sample_count += int(activation.shape[0])
            activation_importance = _normalize_linear_importance(
                activation_energy / max(sample_count, 1)
            )
            activation_quadratic = _block_second_moment(
                smoothed_activations,
                require_independent_samples=True,
            )
            weight_params = _quantize_hierarchical(
                smoothed_weight,
                _ENHANCED_RADIUS,
                activation_importance,
                activation_quadratic,
            )
            weight_params = _refine_weight_block_hessian(
                weight_params,
                smoothed_weight,
                activation_quadratic,
            )
            weight_params = _refine_weight_pair_hessian(
                weight_params,
                smoothed_weight,
                activation_quadratic,
                candidate_coordinates=8,
                rounds=2,
            )
            weight_params = _refine_weight_lv3_hessian(
                weight_params,
                smoothed_weight,
                activation_quadratic,
            )
            weight_params = _refine_weight_block_hessian(
                weight_params,
                smoothed_weight,
                activation_quadratic,
                rounds=2,
            )
            weight_params = _refine_weight_pair_hessian(
                weight_params,
                smoothed_weight,
                activation_quadratic,
                candidate_coordinates=8,
                rounds=1,
            )
            return {
                "weight_params": weight_params,
                "activation_state": {
                    "family": 4,
                    "mode": 0,
                    "radius": _BASE_RADIUS,
                    "permutation": torch.empty(0, dtype=torch.int64),
                    "smooth_balance": smooth_balance,
                    "hadamard_size": _LINEAR_HADAMARD_SIZE,
                    "reflectors": reflectors,
                },
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

        for family in families:
            family_weight = _apply_linear_family(
                weight_ref,
                family,
                permutation,
                smooth_balance,
                inverse_smooth=True,
                hadamard_size=_LINEAR_HADAMARD_SIZE,
            )
            family_activations = [
                _apply_linear_family(
                    activation,
                    family,
                    permutation,
                    smooth_balance,
                    hadamard_size=_LINEAR_HADAMARD_SIZE,
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
            activation_importance = _normalize_linear_importance(
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
            dynamic_importance = activation_importance
            family_weight_params.append(weight_params)
            family_dynamic_importance.append(dynamic_importance)

            for sample_index, activation in enumerate(family_activations):
                activation_routes = _quantize_hierarchical_multi(
                    activation,
                    (_BASE_RADIUS,),
                    weighted_radius=_ENHANCED_RADIUS,
                    importance=dynamic_importance,
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
            "permutation": permutation,
            "smooth_balance": smooth_balance,
            "hadamard_size": _LINEAR_HADAMARD_SIZE,
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
    importance = activation_state.get("importance")
    permutation = activation_state["permutation"]
    smooth_balance = activation_state["smooth_balance"]
    mode = int(activation_state["mode"])
    hadamard_size = int(activation_state.get("hadamard_size", 64))
    with torch.no_grad():
        activation = _dequantize_src(activation_quant, activation_scale)
        activation = _apply_linear_family(
            activation,
            family,
            permutation,
            smooth_balance,
            hadamard_size=hadamard_size,
        )
        if "reflectors" in activation_state:
            activation = _apply_reflectors(
                activation, activation_state["reflectors"]
            )
        return _quantize_hierarchical(
            activation,
            _BASE_RADIUS,
            importance if mode == 1 else None,
        )


def _attention_importance(
    samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    exponent: float = _ATTENTION_BALANCE_EXPONENT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]:
    repeat = q_num_heads // kv_num_heads
    q_energy = torch.zeros((q_num_heads, head_dim), dtype=torch.float32)
    k_energy = torch.zeros((kv_num_heads, head_dim), dtype=torch.float32)
    v_energy = torch.zeros((kv_num_heads, head_dim), dtype=torch.float32)
    q_count = 0
    k_count = 0
    v_count = 0
    balance_patterns = []
    for q, k, v in samples:
        q_heads = q.reshape(q.shape[0], q_num_heads, head_dim)
        k_heads = k.reshape(k.shape[0], kv_num_heads, head_dim)
        v_heads = v.reshape(v.shape[0], kv_num_heads, head_dim)
        sample_q_energy = torch.mean(q_heads.square(), dim=0).reshape(
            kv_num_heads, repeat, head_dim
        ).mean(dim=1)
        sample_k_energy = torch.mean(k_heads.square(), dim=0)
        sample_pattern = exponent * (
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
        v_energy += torch.sum(v_heads.square(), dim=0)
        q_count += int(q.shape[0])
        k_count += int(k.shape[0])
        v_count += int(v.shape[0])
    q_energy = q_energy / max(q_count, 1)
    k_energy = k_energy / max(k_count, 1)
    v_energy = v_energy / max(v_count, 1)
    q_importance = k_energy.repeat_interleave(repeat, dim=0)
    served_q_energy = q_energy.reshape(kv_num_heads, repeat, head_dim).mean(dim=1)
    k_importance = served_q_energy
    permutation = torch.argsort(
        served_q_energy + k_energy, dim=-1, stable=True
    ).to(torch.int64).contiguous()
    log_balance = exponent * (
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
        _normalize_importance(v_energy.flatten()),
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
            v_importance,
        ) = _attention_importance(samples, q_num_heads, kv_num_heads, head_dim)

        # bundle: (q route, k route, v route); route 0=baseline,
        # 1=expanded unweighted, 2=expanded sensitivity-weighted.
        bundles = [
            (0, 0, 0), (0, 0, 1), (0, 0, 2),
            (1, 1, 0), (1, 1, 1), (1, 1, 2),
            (2, 2, 0), (2, 2, 1), (2, 2, 2),
        ]
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
        route_samples = (
            samples
            if len(samples) <= 2
            else (samples[0], samples[-1])
        )
        for q_ref, k_ref, v_ref in route_samples:
            reference_output = _attention_output(
                q_ref, k_ref, v_ref, q_num_heads, kv_num_heads, head_dim
            )
            q_route_map = _quantize_hierarchical_multi(
                q_ref,
                (_BASE_RADIUS, _ATTENTION_QK_RADIUS),
                weighted_radius=_ATTENTION_QK_RADIUS,
                importance=q_importance,
                candidate_offsets=_ATTENTION_SCALE_OFFSETS,
            )
            k_route_map = _quantize_hierarchical_multi(
                k_ref,
                (_BASE_RADIUS, _ATTENTION_QK_RADIUS),
                weighted_radius=_ATTENTION_QK_RADIUS,
                importance=k_importance,
                candidate_offsets=_ATTENTION_SCALE_OFFSETS,
            )
            v_route_map = _quantize_hierarchical_multi(
                v_ref,
                (_BASE_RADIUS, _ATTENTION_RADIUS),
                weighted_radius=_ATTENTION_RADIUS,
                importance=v_importance,
                candidate_offsets=_ATTENTION_V_SCALE_OFFSETS,
            )
            q_routes = [
                q_route_map[("unweighted", _BASE_RADIUS)],
                q_route_map[("unweighted", _ATTENTION_QK_RADIUS)],
                q_route_map[("weighted", _ATTENTION_QK_RADIUS)],
            ]
            k_routes = [
                k_route_map[("unweighted", _BASE_RADIUS)],
                k_route_map[("unweighted", _ATTENTION_QK_RADIUS)],
                k_route_map[("weighted", _ATTENTION_QK_RADIUS)],
            ]
            v_routes = [
                v_route_map[("unweighted", _BASE_RADIUS)],
                v_route_map[("unweighted", _ATTENTION_RADIUS)],
                v_route_map[("weighted", _ATTENTION_RADIUS)],
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
                    _quantize_hierarchical(
                        balanced_q,
                        _ATTENTION_RADIUS,
                        candidate_offsets=_ATTENTION_SCALE_OFFSETS,
                    ),
                    tuple(q_ref.shape),
                )
                balanced_k_value = _dequantize_tgt(
                    _quantize_hierarchical(
                        balanced_k,
                        _ATTENTION_RADIUS,
                        candidate_offsets=_ATTENTION_SCALE_OFFSETS,
                    ),
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
                transformed_q = _fwht(q_ref)
                transformed_k = _fwht(k_ref)
                transformed_q_params = _quantize_hierarchical(
                    transformed_q,
                    _ATTENTION_RADIUS,
                    candidate_offsets=_ATTENTION_SCALE_OFFSETS,
                )
                transformed_k_params = _quantize_hierarchical(
                    transformed_k,
                    _ATTENTION_RADIUS,
                    candidate_offsets=_ATTENTION_SCALE_OFFSETS,
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
                permuted_q = _fwht(
                    _apply_attention_permutation(
                        q_ref,
                        q_num_heads,
                        kv_num_heads,
                        head_dim,
                        permutation,
                    )
                )
                permuted_k = _fwht(
                    _apply_attention_permutation(
                        k_ref,
                        kv_num_heads,
                        kv_num_heads,
                        head_dim,
                        permutation,
                    )
                )
                permuted_q_value = _dequantize_tgt(
                    _quantize_hierarchical(
                        permuted_q,
                        _ATTENTION_RADIUS,
                        candidate_offsets=_ATTENTION_SCALE_OFFSETS,
                    ),
                    tuple(q_ref.shape),
                )
                permuted_k_value = _dequantize_tgt(
                    _quantize_hierarchical(
                        permuted_k,
                        _ATTENTION_RADIUS,
                        candidate_offsets=_ATTENTION_SCALE_OFFSETS,
                    ),
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
                            _fwht(balanced_q),
                            _ATTENTION_RADIUS,
                            candidate_offsets=_ATTENTION_SCALE_OFFSETS,
                        ),
                        tuple(q_ref.shape),
                    )
                    balanced_h_k_value = _dequantize_tgt(
                        _quantize_hierarchical(
                            _fwht(balanced_k),
                            _ATTENTION_RADIUS,
                            candidate_offsets=_ATTENTION_SCALE_OFFSETS,
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
        sequence_rows = int(samples[0][0].shape[0])
        token_hessian = None
        if all(int(q.shape[0]) == sequence_rows for q, _, _ in samples):
            repeat = q_num_heads // kv_num_heads
            token_hessian = torch.zeros(
                (kv_num_heads, sequence_rows, sequence_rows),
                dtype=torch.float32,
            )
            for q_ref, k_ref, _ in samples:
                probabilities = _attention_probabilities(
                    q_ref, k_ref, q_num_heads, kv_num_heads, head_dim
                )
                grouped = probabilities.reshape(
                    kv_num_heads, repeat, sequence_rows, sequence_rows
                )
                token_hessian += torch.einsum(
                    "krqi,krqj->kij", grouped, grouped
                ) / float(max(repeat, 1))
            token_hessian /= float(len(samples))
            diagonal_mean = torch.mean(
                torch.diagonal(token_hessian, dim1=-2, dim2=-1), dim=-1
            ).clamp_min(1.0e-12)
            token_hessian = token_hessian / diagonal_mean[:, None, None]
            token_hessian += 1.0e-4 * torch.eye(
                sequence_rows, dtype=torch.float32
            )
        # Q/K output-Hessian refinements were score-neutral and are omitted.
        # V Hadamard: not currently selected
        q_state = {
            "route": int(q_route),
            "radius": int(_BASE_RADIUS if q_route == 0 else _ATTENTION_QK_RADIUS),
            "importance": q_importance,
            "transform": int(transform),
            "kv_num_heads": int(kv_num_heads),
            "permutation": permutation,
            "balance": balance,
        }
        k_state = {
            "route": int(k_route),
            "radius": int(_BASE_RADIUS if k_route == 0 else _ATTENTION_QK_RADIUS),
            "importance": k_importance,
            "transform": int(transform),
            "kv_num_heads": int(kv_num_heads),
            "permutation": permutation,
            "balance": balance,
        }
        v_state = {
            "route": int(v_route),
            "radius": int(_BASE_RADIUS if v_route == 0 else _ATTENTION_RADIUS),
            "importance": v_importance if v_route == 2 else None,
            "transform": 0,
            "v_offsets": _ATTENTION_V_SCALE_OFFSETS,
            "v_hadamard": False,
            "token_hessian": (
                token_hessian.to(torch.float16)
                if token_hessian is not None
                else None
            ),
        }
        return {"q_state": q_state, "k_state": k_state, "v_state": v_state}


def _prepare_dynamic_attention_value(
    value: torch.Tensor,
    num_heads: int,
    head_dim: int,
    state: Any,
    stream: str,
) -> torch.Tensor:
    transform = int(state.get("transform", 0))
    if stream == "v" and state.get("v_hadamard", False) and head_dim % 64 == 0:
        value = _fwht(value)
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
            value = _fwht(value)
    elif transform != 0:
        raise ValueError("unknown attention transform")
    if stream == "k":
        # K -> K - mean_token(K) shifts every logit in a query row by the
        # same scalar, so softmax is unchanged before quantization.
        value = value - torch.mean(value, dim=0, keepdim=True)
    return value


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
    importance = state.get("importance") if route == 2 else None
    with torch.no_grad():
        value = _prepare_dynamic_attention_value(
            _dequantize_src(quant, scale),
            num_heads,
            head_dim,
            state,
            stream,
        )
        return _quantize_hierarchical(
            value,
            radius,
            importance,
            candidate_offsets=(
                tuple(state["v_offsets"])
                if stream == "v" and radius == _ATTENTION_RADIUS
                else (_ATTENTION_SCALE_OFFSETS
                      if radius == _ATTENTION_RADIUS
                      else None)
            ),
        )


def _refine_q_output_hessian(
    params: dict[str, torch.Tensor],
    target: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    output_hessian: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    if output_hessian is None or tuple(output_hessian.shape) != (
        q_num_heads, head_dim, head_dim
    ):
        return params
    sequence_rows = int(target.shape[0])
    refined = {key: value.clone() for key, value in params.items()}
    reconstructed = _dequantize_tgt(refined, tuple(target.shape))
    error = (reconstructed - target).reshape(
        sequence_rows, q_num_heads, head_dim
    )
    hessian = 0.5 * output_hessian.to(torch.float32) + 0.5 * torch.eye(
        head_dim, dtype=torch.float32
    ).unsqueeze(0)
    mantissa = refined["mant"].reshape(
        sequence_rows, q_num_heads, head_dim
    )
    sign = refined["sign"].reshape(sequence_rows, q_num_heads, head_dim)
    effective_scale = (
        refined["scale_factor"]
        * refined["scale_lv2"]
        * refined["scale_lv3"]
    ).expand_as(refined["mant"]).reshape(
        sequence_rows, q_num_heads, head_dim
    )
    target_sign = torch.sign(target).reshape(
        sequence_rows, q_num_heads, head_dim
    )
    curvature = torch.diagonal(hessian, dim1=-2, dim2=-1).unsqueeze(0)
    infinity = torch.tensor(float("inf"), dtype=torch.float32)
    for _ in range(0):
        gradient = torch.einsum("hij,shj->shi", hessian, error)
        step = _MANTISSA_STEP * effective_scale
        increment_sign = torch.where(sign != 0.0, sign, target_sign)
        increment = increment_sign * step
        decrement = -sign * step
        increment_change = torch.where(
            mantissa < _MANTISSA_MAX,
            2.0 * increment * gradient + increment.square() * curvature,
            infinity,
        )
        decrement_change = torch.where(
            mantissa > 0.0,
            2.0 * decrement * gradient + decrement.square() * curvature,
            infinity,
        )
        use_increment = increment_change <= decrement_change
        best_change = torch.where(
            use_increment, increment_change, decrement_change
        )
        best_value, best_index = torch.min(best_change, dim=-1)
        selected = torch.zeros_like(best_change, dtype=torch.bool)
        selected.scatter_(-1, best_index.unsqueeze(-1), True)
        selected &= best_value.unsqueeze(-1) < 0.0
        delta = torch.where(
            selected,
            torch.where(use_increment, increment, decrement),
            0.0,
        )
        candidate_mantissa = torch.where(
            use_increment,
            mantissa + _MANTISSA_STEP,
            mantissa - _MANTISSA_STEP,
        )
        mantissa.copy_(torch.where(selected, candidate_mantissa, mantissa))
        candidate_sign = torch.where(
            candidate_mantissa == 0.0,
            0.0,
            torch.where(
                use_increment & (sign == 0.0), increment_sign, sign
            ),
        )
        sign.copy_(torch.where(selected, candidate_sign, sign))
        error += delta
    return {key: value.contiguous() for key, value in refined.items()}


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


def _refine_k_output_hessian(
    params: dict[str, torch.Tensor],
    target: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    output_hessian: torch.Tensor | None,
    output_factors: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    sequence_rows = int(target.shape[0])
    if (
        output_hessian is None
        or output_factors is None
        or tuple(output_hessian.shape)
        != (kv_num_heads, sequence_rows, head_dim, head_dim)
        or tuple(output_factors.shape[:3])
        != (kv_num_heads, sequence_rows, head_dim)
    ):
        return params
    refined = {key: value.clone() for key, value in params.items()}
    reconstructed = _dequantize_tgt(refined, tuple(target.shape))
    error = (reconstructed - target).reshape(
        sequence_rows, kv_num_heads, head_dim
    )
    hessian = 0.35 * output_hessian.to(torch.float32) + 0.65 * torch.eye(
        head_dim, dtype=torch.float32
    ).reshape(1, 1, head_dim, head_dim)
    factors = output_factors.to(torch.float32)
    gamma = 0.15
    mantissa = refined["mant"].reshape(
        sequence_rows, kv_num_heads, head_dim
    )
    sign = refined["sign"].reshape(sequence_rows, kv_num_heads, head_dim)
    effective_scale = (
        refined["scale_factor"]
        * refined["scale_lv2"]
        * refined["scale_lv3"]
    ).expand_as(refined["mant"]).reshape(
        sequence_rows, kv_num_heads, head_dim
    )
    target_sign = torch.sign(target).reshape(
        sequence_rows, kv_num_heads, head_dim
    )
    curvature = (
        torch.diagonal(hessian, dim1=-2, dim2=-1)
        + gamma * torch.sum(factors.square(), dim=-1)
    ).permute(1, 0, 2)
    projected_error = torch.einsum("shd,hsdr->hr", error, factors)
    infinity = torch.tensor(float("inf"), dtype=torch.float32)
    for _ in range(0):
        local_gradient = torch.einsum("hsij,shj->shi", hessian, error)
        gradient = local_gradient + gamma * torch.einsum(
            "hr,hsdr->shd", projected_error, factors
        )
        step = _MANTISSA_STEP * effective_scale
        increment_sign = torch.where(sign != 0.0, sign, target_sign)
        increment = increment_sign * step
        decrement = -sign * step
        increment_change = torch.where(
            mantissa < _MANTISSA_MAX,
            2.0 * increment * gradient + increment.square() * curvature,
            infinity,
        )
        decrement_change = torch.where(
            mantissa > 0.0,
            2.0 * decrement * gradient + decrement.square() * curvature,
            infinity,
        )
        use_increment = increment_change <= decrement_change
        best_change = torch.where(
            use_increment, increment_change, decrement_change
        )
        best_value, best_index = torch.min(best_change, dim=-1)
        selected = torch.zeros_like(best_change, dtype=torch.bool)
        selected.scatter_(-1, best_index.unsqueeze(-1), True)
        selected &= best_value.unsqueeze(-1) < 0.0
        delta = torch.where(
            selected,
            torch.where(use_increment, increment, decrement),
            0.0,
        )
        candidate_error = error + delta
        candidate_projected = projected_error + torch.einsum(
            "shd,hsdr->hr", delta, factors
        )
        old_objective = (
            torch.einsum("shd,shd->h", error, local_gradient)
            + gamma * torch.sum(projected_error.square(), dim=-1)
        )
        new_objective = (
            torch.einsum(
                "shd,hsde,she->h", candidate_error, hessian, candidate_error
            )
            + gamma * torch.sum(candidate_projected.square(), dim=-1)
        )
        accepted = selected & (new_objective < old_objective)[None, :, None]
        delta = torch.where(accepted, delta, 0.0)
        candidate_mantissa = torch.where(
            use_increment,
            mantissa + _MANTISSA_STEP,
            mantissa - _MANTISSA_STEP,
        )
        mantissa.copy_(torch.where(accepted, candidate_mantissa, mantissa))
        candidate_sign = torch.where(
            candidate_mantissa == 0.0,
            0.0,
            torch.where(
                use_increment & (sign == 0.0), increment_sign, sign
            ),
        )
        sign.copy_(torch.where(accepted, candidate_sign, sign))
        error += delta
        projected_error += torch.einsum("shd,hsdr->hr", delta, factors)
    return {key: value.contiguous() for key, value in refined.items()}


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


def _refine_v_token_hessian(
    params: dict[str, torch.Tensor],
    source: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    hessian: torch.Tensor,
    rounds: int = 4,
) -> dict[str, torch.Tensor]:
    if rounds <= 0:
        return params
    sequence_rows = int(source.shape[0])
    if tuple(hessian.shape) != (kv_num_heads, sequence_rows, sequence_rows):
        return params
    refined = {key: value.clone() for key, value in params.items()}
    reconstructed = _dequantize_tgt(refined, tuple(source.shape))
    error = (reconstructed - source).reshape(
        sequence_rows, kv_num_heads, head_dim
    )
    mantissa = refined["mant"].reshape(
        sequence_rows, kv_num_heads, head_dim
    )
    sign = refined["sign"].reshape(sequence_rows, kv_num_heads, head_dim)
    effective_scale = (
        refined["scale_factor"]
        * refined["scale_lv2"]
        * refined["scale_lv3"]
    ).expand_as(refined["mant"]).reshape(
        sequence_rows, kv_num_heads, head_dim
    )
    target_sign = torch.sign(source).reshape(
        sequence_rows, kv_num_heads, head_dim
    )
    infinity = torch.tensor(float("inf"), dtype=torch.float32)
    for _ in range(rounds):
        gradients = torch.einsum("hrs,shd->rhd", hessian, error)
        for row in range(sequence_rows):
            gradient = gradients[row]
            current_mantissa = mantissa[row]
            current_sign = sign[row]
            step = _MANTISSA_STEP * effective_scale[row]
            increment_sign = torch.where(
                current_sign != 0.0, current_sign, target_sign[row]
            )
            increment = increment_sign * step
            decrement = -current_sign * step
            curvature = hessian[:, row, row].unsqueeze(-1)
            increment_change = torch.where(
                current_mantissa < _MANTISSA_MAX,
                2.0 * increment * gradient + increment.square() * curvature,
                infinity,
            )
            decrement_change = torch.where(
                current_mantissa > 0.0,
                2.0 * decrement * gradient + decrement.square() * curvature,
                infinity,
            )
            use_increment = increment_change <= decrement_change
            best_change = torch.where(
                use_increment, increment_change, decrement_change
            )
            accepted = best_change < 0.0
            delta = torch.where(
                accepted,
                torch.where(use_increment, increment, decrement),
                0.0,
            )
            new_mantissa = torch.where(
                accepted,
                torch.where(
                    use_increment,
                    current_mantissa + _MANTISSA_STEP,
                    current_mantissa - _MANTISSA_STEP,
                ),
                current_mantissa,
            )
            mantissa[row] = new_mantissa
            sign[row] = torch.where(
                new_mantissa == 0.0,
                0.0,
                torch.where(
                    accepted & use_increment & (current_sign == 0.0),
                    increment_sign,
                    current_sign,
                ),
            )
            error[row] += delta
            gradients += (
                hessian[:, :, row].transpose(0, 1).unsqueeze(-1)
                * delta.unsqueeze(0)
            )
    return {key: value.contiguous() for key, value in refined.items()}


def _refine_v_pair_token_hessian(
    params: dict[str, torch.Tensor],
    source: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    hessian: torch.Tensor,
    candidate_tokens: int = 4,
    rounds: int = 1,
) -> dict[str, torch.Tensor]:
    """Escape V token-coordinate minima with exact two-token moves."""
    sequence_rows = int(source.shape[0])
    if (
        rounds <= 0
        or candidate_tokens < 2
        or tuple(hessian.shape)
        != (kv_num_heads, sequence_rows, sequence_rows)
    ):
        return params
    token_count = min(candidate_tokens, sequence_rows)
    move_count = 2 * token_count
    refined = {key: value.clone() for key, value in params.items()}
    reconstructed = _dequantize_tgt(refined, tuple(source.shape))
    source_heads = source.reshape(sequence_rows, kv_num_heads, head_dim)
    error = (reconstructed - source).reshape_as(source_heads)
    mantissa = refined["mant"].reshape_as(source_heads)
    sign = refined["sign"].reshape_as(source_heads)
    effective_scale = (
        refined["scale_factor"]
        * refined["scale_lv2"]
        * refined["scale_lv3"]
    ).expand_as(refined["mant"]).reshape_as(source_heads)
    target_sign = torch.sign(source_heads)
    infinity = torch.tensor(float("inf"), dtype=torch.float32)
    upper_triangle = torch.triu(
        torch.ones((move_count, move_count), dtype=torch.bool), diagonal=1
    )

    for _ in range(rounds):
        gradient = torch.einsum("hst,thd->shd", hessian, error)
        step = _MANTISSA_STEP * effective_scale
        increment_sign = torch.where(sign != 0.0, sign, target_sign)
        increments = torch.stack((increment_sign * step, -sign * step), dim=-1)
        valid_moves = torch.stack(
            (mantissa < _MANTISSA_MAX, mantissa > 0.0), dim=-1
        )
        curvature = torch.diagonal(hessian, dim1=-2, dim2=-1)
        curvature = curvature.permute(1, 0).unsqueeze(-1).unsqueeze(-1)
        single_change = (
            2.0 * increments * gradient.unsqueeze(-1)
            + increments.square() * curvature
        )
        eligible_change = torch.where(
            valid_moves & (single_change >= 0.0), single_change, infinity
        ).permute(1, 2, 0, 3)
        move_delta = increments.permute(1, 2, 0, 3)
        coordinate_score = torch.min(eligible_change, dim=-1).values
        _, token_index = torch.topk(
            coordinate_score, k=token_count, dim=-1, largest=False
        )
        gather_token = token_index.unsqueeze(-1).expand(
            kv_num_heads, head_dim, token_count, 2
        )
        selected_change = torch.gather(
            eligible_change, 2, gather_token
        ).reshape(kv_num_heads, head_dim, move_count)
        selected_delta = torch.gather(
            move_delta, 2, gather_token
        ).reshape(kv_num_heads, head_dim, move_count)
        selected_index = token_index.unsqueeze(-1).expand(
            kv_num_heads, head_dim, token_count, 2
        ).reshape(kv_num_heads, head_dim, move_count)

        expanded_hessian = hessian.unsqueeze(1).expand(
            kv_num_heads, head_dim, sequence_rows, sequence_rows
        )
        selected_rows = torch.gather(
            expanded_hessian,
            2,
            selected_index.unsqueeze(-1).expand(-1, -1, -1, sequence_rows),
        )
        selected_cross = torch.gather(
            selected_rows,
            3,
            selected_index.unsqueeze(-2).expand(-1, -1, move_count, -1),
        )
        pair_change = (
            selected_change.unsqueeze(-1)
            + selected_change.unsqueeze(-2)
            + 2.0
            * selected_delta.unsqueeze(-1)
            * selected_delta.unsqueeze(-2)
            * selected_cross
        )
        valid_pair = (
            upper_triangle.reshape(1, 1, move_count, move_count)
            & (selected_index.unsqueeze(-1) != selected_index.unsqueeze(-2))
            & torch.isfinite(pair_change)
        )
        pair_change = torch.where(valid_pair, pair_change, infinity)
        best_change, flat_pair_index = torch.min(
            pair_change.reshape(kv_num_heads, head_dim, -1), dim=-1
        )
        first_move = torch.div(
            flat_pair_index, move_count, rounding_mode="floor"
        )
        second_move = torch.remainder(flat_pair_index, move_count)
        accepted = best_change < 0.0
        first_token = torch.gather(
            selected_index, -1, first_move.unsqueeze(-1)
        ).squeeze(-1)
        second_token = torch.gather(
            selected_index, -1, second_move.unsqueeze(-1)
        ).squeeze(-1)
        first_delta = torch.gather(
            selected_delta, -1, first_move.unsqueeze(-1)
        ).squeeze(-1)
        second_delta = torch.gather(
            selected_delta, -1, second_move.unsqueeze(-1)
        ).squeeze(-1)
        pair_delta = torch.zeros(
            (kv_num_heads, head_dim, sequence_rows), dtype=torch.float32
        )
        pair_delta.scatter_add_(
            -1,
            first_token.unsqueeze(-1),
            torch.where(accepted, first_delta, 0.0).unsqueeze(-1),
        )
        pair_delta.scatter_add_(
            -1,
            second_token.unsqueeze(-1),
            torch.where(accepted, second_delta, 0.0).unsqueeze(-1),
        )
        pair_delta = pair_delta.permute(2, 0, 1)
        reconstructed_heads = sign * mantissa * effective_scale + pair_delta
        mantissa.copy_(
            torch.clamp(
                torch.abs(reconstructed_heads) / effective_scale,
                min=0.0,
                max=_MANTISSA_MAX,
            )
        )
        sign.copy_(
            torch.where(
                mantissa == 0.0, 0.0, torch.sign(reconstructed_heads)
            )
        )
        error += pair_delta
    return {key: value.contiguous() for key, value in refined.items()}


def _refine_v_lv3_token_hessian(
    params: dict[str, torch.Tensor],
    source: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    hessian: torch.Tensor,
    rounds: int = 1,
) -> dict[str, torch.Tensor]:
    """Flip V four-value subtrees with sequential exact token-Hessian moves."""
    sequence_rows = int(source.shape[0])
    if (
        rounds <= 0
        or head_dim % 64 != 0
        or tuple(hessian.shape)
        != (kv_num_heads, sequence_rows, sequence_rows)
    ):
        return params
    blocks_per_head = head_dim // 64
    refined = {key: value.clone() for key, value in params.items()}
    reconstructed = _dequantize_tgt(refined, tuple(source.shape))
    source_quartets = source.reshape(
        sequence_rows, kv_num_heads, blocks_per_head, 16, 4
    )
    error = (reconstructed - source).reshape_as(source_quartets)
    mantissa = refined["mant"].reshape_as(source_quartets)
    sign = refined["sign"].reshape_as(source_quartets)
    scale_factor = refined["scale_factor"].reshape(
        sequence_rows, kv_num_heads, blocks_per_head, 1, 1
    )
    lv2 = refined["scale_lv2"].expand(
        sequence_rows, kv_num_heads * blocks_per_head, 8, 2, 1
    ).reshape(sequence_rows, kv_num_heads, blocks_per_head, 16, 1)
    lv3 = refined["scale_lv3"].reshape(
        sequence_rows, kv_num_heads, blocks_per_head, 16, 1
    )

    for _ in range(rounds):
        flat_error = error.reshape(sequence_rows, kv_num_heads, head_dim)
        gradients = torch.einsum("hrs,shd->rhd", hessian, flat_error)
        for row in range(sequence_rows):
            gradient = gradients[row].reshape(
                kv_num_heads, blocks_per_head, 16, 4
            )
            candidate_lv3 = 3.0 - lv3[row]
            effective_scale = (
                scale_factor[row] * lv2[row] * candidate_lv3
            )
            row_source = source_quartets[row]
            candidate_mantissa = torch.clamp(
                torch.round(
                    torch.abs(row_source)
                    / effective_scale
                    / _MANTISSA_STEP
                )
                * _MANTISSA_STEP,
                min=0.0,
                max=_MANTISSA_MAX,
            )
            candidate_value = (
                torch.sign(row_source)
                * candidate_mantissa
                * effective_scale
            )
            current_value = row_source + error[row]
            delta = candidate_value - current_value
            curvature = hessian[:, row, row].reshape(
                kv_num_heads, 1, 1
            )
            objective_change = (
                2.0 * torch.sum(delta * gradient, dim=-1)
                + curvature * torch.sum(delta.square(), dim=-1)
            )
            best_change, best_quartet = torch.min(
                objective_change, dim=-1
            )
            selected = torch.zeros_like(objective_change, dtype=torch.bool)
            selected.scatter_(-1, best_quartet.unsqueeze(-1), True)
            selected &= best_change.unsqueeze(-1) < 0.0
            selected_values = selected.unsqueeze(-1)
            lv3[row].copy_(
                torch.where(selected_values, candidate_lv3, lv3[row])
            )
            mantissa[row].copy_(
                torch.where(
                    selected_values, candidate_mantissa, mantissa[row]
                )
            )
            sign[row].copy_(
                torch.where(
                    selected_values,
                    torch.where(
                        candidate_mantissa == 0.0,
                        0.0,
                        torch.sign(row_source),
                    ),
                    sign[row],
                )
            )
            accepted_delta = torch.where(selected_values, delta, 0.0)
            error[row] += accepted_delta
            gradients += (
                hessian[:, :, row].transpose(0, 1).unsqueeze(-1)
                * accepted_delta.reshape(kv_num_heads, head_dim).unsqueeze(0)
            )
    return {key: value.contiguous() for key, value in refined.items()}


def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    params = _dynamic_attention(
        v_quant, v_scale, kv_num_heads, head_dim, v_state, "v"
    )
    token_hessian = v_state.get("token_hessian")
    if token_hessian is None:
        return params
    with torch.no_grad():
        source = _prepare_dynamic_attention_value(
            _dequantize_src(v_quant, v_scale),
            kv_num_heads,
            head_dim,
            v_state,
            "v",
        )
        sequence_rows = int(source.shape[0])
        if tuple(token_hessian.shape) != (
            kv_num_heads, sequence_rows, sequence_rows
        ):
            return params
        hessian_weight = float(v_state.get("token_hessian_weight", 0.7))
        refine_rounds = int(v_state.get("token_hessian_rounds", 4))
        hessian = (
            hessian_weight * token_hessian.to(torch.float32)
            + (1.0 - hessian_weight)
            * torch.eye(sequence_rows, dtype=torch.float32).unsqueeze(0)
        )
        refined = _refine_v_token_hessian(
            params,
            source,
            kv_num_heads,
            head_dim,
            hessian,
            rounds=refine_rounds,
        )
        refined = _refine_v_pair_token_hessian(
            refined,
            source,
            kv_num_heads,
            head_dim,
            hessian,
        )
        refined = _refine_v_lv3_token_hessian(
            refined,
            source,
            kv_num_heads,
            head_dim,
            hessian,
        )
        refined = _refine_v_token_hessian(
            refined,
            source,
            kv_num_heads,
            head_dim,
            hessian,
            rounds=2,
        )
        return _refine_v_pair_token_hessian(
            refined,
            source,
            kv_num_heads,
            head_dim,
            hessian,
        )
