"""NVFP4 -> HiF4 converter for the 2026 Huawei Algorithm Contest.

Version 14:
- Use v138's output-aware Linear calibration, including the shared Hadamard
  transform, ridge-fitted weight candidates, and held-out code refinement.
- Keep v13's proven Attention path unchanged: shared Q/K Hadamard rotation
  with block size 16 and per-block E6M2 candidate search for Q/K/V.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch


E6M2_MIN = 2.0 ** (-48)
E6M2_MAX = 49152.0  # 2^15 * 1.5

_LINEAR_SCALE_OFFSETS = (0, 1, 2, 3, 4)
_LINEAR_HADAMARD = 64
_LINEAR_BALANCE_EXPONENT = 0.39
_LINEAR_RIDGE_LAMBDAS = (10.0, 20.0)
_LINEAR_RIDGE_SWITCH_MARGIN = 0.05
_WEIGHT_REFINEMENT_RIDGE = 0.01
_WEIGHT_REFINEMENT_STEP = 0.04
_WEIGHT_REFINEMENT_CHECKPOINTS = (1, 2, 3, 4, 5)


# =============================================================================
# NVFP4 helper
# =============================================================================

def dequantize_nvfp4(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    blk_size: int = 16,
) -> torch.Tensor:
    """Decode NVFP4 carrier + block scale into a BF16 tensor."""
    channels = int(quant_float.shape[-1])
    if channels % blk_size != 0:
        raise ValueError(
            f"last dimension {channels} is not divisible by block size {blk_size}"
        )

    x = quant_float.unflatten(-1, (-1, blk_size))
    x = x * scale_float.unsqueeze(-1)
    return x.flatten(-2, -1).to(torch.bfloat16)


# =============================================================================
# HiF4 conversion helpers
# =============================================================================

def _e6m2_quant(x: torch.Tensor) -> torch.Tensor:
    """Round a positive scale into the legal unsigned E6M2 value set."""
    x = torch.clamp(x, min=E6M2_MIN, max=E6M2_MAX)
    exp = torch.floor(torch.log2(x))
    mant = torch.round(x * (2.0 ** (2.0 - exp)))
    return mant * (2.0 ** (exp - 2.0))


def _s1p2_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a signed value into HiF4 S1P2 sign and mantissa fields."""
    sign = torch.sign(x)
    mag = torch.abs(x)
    q = torch.round(mag * 4.0) / 4.0
    q = torch.clamp(q, min=0.0, max=1.75)
    sign = torch.where(q == 0.0, torch.zeros_like(sign), sign)
    return sign, q


def _e6m2_candidates(sf: torch.Tensor, radius: int = 1) -> list[torch.Tensor]:
    """Return a small list of legal E6M2 candidates around ``sf``."""
    base_exp = torch.floor(torch.log2(sf))
    candidates: list[torch.Tensor] = []
    for delta in range(-radius, radius + 1):
        exp = (base_exp + delta).clamp(-48.0, 15.0)
        for mant in (1.0, 1.25, 1.5, 1.75):
            value = (2.0 ** exp) * mant
            value = torch.clamp(value, min=E6M2_MIN, max=E6M2_MAX)
            candidates.append(value)
    return candidates


def _hif4_from_scale(
    xb: torch.Tensor,
    v16: torch.Tensor,
    v8: torch.Tensor,
    scale_factor: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Quantize a reshaped block tensor given a candidate E6M2 scale factor."""
    rec = 1.0 / scale_factor

    e1_8 = (v8 * rec.unsqueeze(-1) >= 4.0).to(torch.float32)
    e1_8_pairs = e1_8.unsqueeze(-1)
    e1_16 = (
        v16
        * rec.unsqueeze(-1).unsqueeze(-1)
        * (2.0 ** (-e1_8_pairs))
        >= 2.0
    ).to(torch.float32)

    e1_8_for = e1_8.unsqueeze(-1).unsqueeze(-1)
    e1_16_for = e1_16.unsqueeze(-1)
    element_scale = (
        scale_factor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        * (2.0 ** e1_8_for)
        * (2.0 ** e1_16_for)
    )
    scaled = xb / element_scale
    sign, mant = _s1p2_quant(scaled)
    dequant = sign * mant * element_scale

    params = {
        "scale_factor": scale_factor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
        "scale_lv2": (2.0 ** e1_8).unsqueeze(-1).unsqueeze(-1),
        "scale_lv3": (2.0 ** e1_16).unsqueeze(-1),
        "sign": sign,
        "mant": mant,
    }
    return params, dequant


def _quantize_hif4(
    x: torch.Tensor,
    importance: torch.Tensor | None = None,
    search_candidates: bool = False,
    candidate_radius: int = 1,
) -> dict[str, torch.Tensor]:
    """Convert a BF16/FP32 tensor to HiF4 parameters along its last dim."""
    x = x.to(torch.float32)
    shape = x.shape
    channels = int(shape[-1])
    if channels % 64 != 0:
        raise ValueError(
            f"last dimension {channels} is not divisible by HiF4 block size 64"
        )

    num_blocks = channels // 64
    prefix = shape[:-1]
    xb = x.reshape(*prefix, num_blocks, 8, 2, 4)

    v16 = xb.abs().amax(dim=-1)
    v8 = v16.amax(dim=-1)
    vmax = v8.amax(dim=-1, keepdim=True)
    sf_std = _e6m2_quant(vmax.squeeze(-1) / 7.0)

    if not search_candidates:
        params, _ = _hif4_from_scale(xb, v16, v8, sf_std)
        return params

    candidates = _e6m2_candidates(sf_std, radius=candidate_radius)
    all_params: list[dict[str, torch.Tensor]] = []
    costs: list[torch.Tensor] = []

    if importance is not None:
        imp_b = importance.to(torch.float32)
    else:
        imp_b = None

    for sf in candidates:
        params, dequant = _hif4_from_scale(xb, v16, v8, sf)
        err = (dequant - xb) ** 2
        if imp_b is not None:
            cost = (err * imp_b).sum(dim=(-3, -2, -1))
        else:
            cost = err.sum(dim=(-3, -2, -1))
        all_params.append(params)
        costs.append(cost)

    cost_stack = torch.stack(costs, dim=0)  # [C, *prefix, nblocks]
    best_idx = torch.argmin(cost_stack, dim=0)  # [*prefix, nblocks]

    result: dict[str, torch.Tensor] = {}
    for key in all_params[0]:
        stacked = torch.stack([p[key] for p in all_params], dim=0)
        tail_shape = stacked.shape[1 + len(prefix) + 1:]
        index = best_idx.reshape((1,) + best_idx.shape + (1,) * len(tail_shape))
        index = index.expand((1,) + best_idx.shape + tail_shape)
        result[key] = torch.gather(stacked, 0, index).squeeze(0)
    return result


def _dequantize_hif4_params(
    params: dict[str, torch.Tensor],
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Decode HiF4 parameters back to a floating-point tensor."""
    return (
        params["sign"].to(torch.float32)
        * params["mant"].to(torch.float32)
        * params["scale_lv3"].to(torch.float32)
        * params["scale_lv2"].to(torch.float32)
        * params["scale_factor"].to(torch.float32)
    ).reshape(shape)


def _decode_e6m2(code: torch.Tensor) -> torch.Tensor:
    """Decode the compact unsigned E6M2 code used by the Linear search."""
    code64 = code.to(torch.int64)
    mantissa = 1.0 + (code64 & 3).to(torch.float32) / 4.0
    return torch.ldexp(mantissa, (code64 >> 2) - 48)


def _encode_e6m2(scale: torch.Tensor) -> torch.Tensor:
    """Encode a positive scale as a compact legal unsigned E6M2 code."""
    positive = scale > 0
    safe = torch.where(positive, scale, torch.ones_like(scale))
    exponent = torch.floor(torch.log2(safe)).to(torch.int64)
    normalized = torch.ldexp(safe, -exponent)
    mantissa = torch.round((normalized - 1.0) * 4.0).to(torch.int64)
    carry = mantissa == 4
    exponent = (exponent + carry.to(torch.int64)).clamp(-48, 15)
    mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa).clamp(0, 3)
    code = (((exponent + 48) << 2) | mantissa).clamp(0, 0xFE)
    default = torch.full_like(code, 48 << 2)
    return torch.where(positive, code, default).to(torch.uint8)


def _best_linear_for_base(
    groups: torch.Tensor,
    scale_code: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Find the best legal E1_8/E1_16 split for one Linear base scale."""
    base = _decode_e6m2(scale_code)
    factors = groups.new_tensor((1.0, 2.0, 4.0)).view(
        *([1] * (groups.ndim - 2)), 3, 1, 1
    )
    scaled = groups.unsqueeze(-3) / base.unsqueeze(-3) / factors
    negative = scaled < 0
    mantissa = (
        torch.round(scaled.abs().clamp(max=1.75) * 4.0)
        .to(torch.uint8)
        .clamp(max=7)
    )
    reconstructed = (
        base.unsqueeze(-3)
        * factors
        * torch.where(negative, -1.0, 1.0)
        * mantissa.to(torch.float32)
        / 4.0
    )
    errors = (groups.unsqueeze(-3) - reconstructed).square()
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
    selected = torch.gather(reconstructed, -3, index).squeeze(-3)
    selected_errors = (groups - selected).square()
    if weights is not None:
        selected_errors = selected_errors * weights
    score = selected_errors.sum(dim=(-3, -2, -1))
    return (
        score,
        scale_code,
        lv2.unsqueeze(-1).unsqueeze(-1),
        lv3.unsqueeze(-1),
        torch.gather(negative, -3, index).squeeze(-3),
        torch.gather(mantissa, -3, index).squeeze(-3),
    )


def _quantize_linear_values(
    values: torch.Tensor,
    importance: torch.Tensor | None = None,
    offsets: Iterable[int] = _LINEAR_SCALE_OFFSETS,
) -> dict[str, torch.Tensor]:
    """Quantize Linear values using v138's compact per-block scale search."""
    channels = int(values.shape[-1])
    groups = values.to(torch.float32).reshape(
        *values.shape[:-1], channels // 64, 8, 2, 4
    )
    weights = None
    if importance is not None:
        weights = importance.to(torch.float32).reshape(
            1, channels // 64, 8, 2, 4
        )
    peak = groups.abs().amax(dim=(-3, -2, -1), keepdim=True) / 7.0
    peak_code = _encode_e6m2(peak)
    best = None
    for offset in offsets:
        code = (
            peak_code.to(torch.int16).add(int(offset)).clamp(0, 0xFE).to(torch.uint8)
        )
        current = _best_linear_for_base(groups, code, weights)
        if best is None:
            best = current
            continue
        choose = current[0] < best[0]
        mask = choose[..., None, None, None]
        best = (torch.where(choose, current[0], best[0]),) + tuple(
            torch.where(mask, new, old)
            for new, old in zip(current[1:], best[1:])
        )

    if best is None:
        raise RuntimeError("Linear scale candidate set must not be empty")
    _, code, lv2, lv3, negative, mantissa_code = best
    mantissa = mantissa_code.to(torch.bfloat16) / 4.0
    sign = torch.where(
        mantissa == 0,
        torch.zeros_like(mantissa),
        torch.where(
            negative,
            -torch.ones_like(mantissa),
            torch.ones_like(mantissa),
        ),
    )
    return {
        "scale_factor": _decode_e6m2(code).to(torch.bfloat16),
        "scale_lv2": 1.0 + lv2.to(torch.bfloat16),
        "scale_lv3": 1.0 + lv3.to(torch.bfloat16),
        "sign": sign,
        "mant": mantissa,
    }


def _dequantize_linear_params(params: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        params["scale_factor"].to(torch.float32)
        * params["scale_lv2"].to(torch.float32)
        * params["scale_lv3"].to(torch.float32)
        * params["sign"].to(torch.float32)
        * params["mant"].to(torch.float32)
    ).flatten(-4, -1)


def _choose_weight_rows(
    target_output: torch.Tensor,
    quantized_activation: torch.Tensor,
    candidates: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Select the lowest-output-MSE HiF4 candidate per output channel."""
    weights = torch.stack([_dequantize_linear_params(p) for p in candidates])
    outputs = torch.matmul(quantized_activation, weights.transpose(1, 2))
    choices = (target_output.unsqueeze(0) - outputs).square().mean(1).argmin(0)
    rows = torch.arange(choices.numel())
    return {
        name: torch.stack([params[name] for params in candidates])[choices, rows]
        for name in candidates[0]
    }


def _output_aware_quantize_weight(
    activation: torch.Tensor,
    weight: torch.Tensor,
    baseline: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Fit a fixed HiF4 Q(W) directly against calibration output A @ W.T."""
    activation_params = _quantize_linear_values(activation)
    quantized_activation = _dequantize_linear_params(activation_params)
    activation_importance = activation.square().mean(0)
    candidates = [baseline, _quantize_linear_values(weight)]
    target_output = activation @ weight.T
    initial_weights = torch.stack(
        [_dequantize_linear_params(params) for params in candidates]
    )
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
            _quantize_linear_values(
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
            kernel + ridge_lambda * ridge_scale * identity,
            residual,
        )
        fitted_weight = weight + (quantized_activation.T @ dual).T
        ridge_candidates.append(
            _quantize_linear_values(fitted_weight, activation_importance)
        )

    row_params = [
        _choose_weight_rows(
            target_output,
            quantized_activation,
            candidates + [ridge_candidate],
        )
        for ridge_candidate in ridge_candidates
    ]
    row_errors = [
        (
            target_output
            - quantized_activation @ _dequantize_linear_params(params).T
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
    """Refine fixed weight codes on train calibration data and select held-out."""
    if len(activation_parts) < 2:
        return weight_params
    quantized_parts = [
        _dequantize_linear_params(_quantize_linear_values(part))
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
        candidate_weight = _dequantize_linear_params(candidate)
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
        _quantize_linear_values(continuous_weight, train_quantized.square().mean(0))
    )
    heldout_target = activation_parts[-1] @ weight.T
    heldout_quantized = quantized_parts[-1]
    return _choose_weight_rows(heldout_target, heldout_quantized, checkpoints)


def _rms(values: torch.Tensor) -> torch.Tensor:
    return (
        values.to(torch.float32)
        .reshape(-1, values.shape[-1])
        .square()
        .mean(0)
        .sqrt()
    )


def _linear_multiplier(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    tiny = torch.finfo(torch.float32).tiny
    multiplier = (right.clamp_min(tiny) / left.clamp_min(tiny)).pow(
        _LINEAR_BALANCE_EXPONENT
    )
    multiplier = multiplier.clamp(1.0 / 16.0, 16.0)
    return torch.where(
        (left > 0) & (right > 0), multiplier, torch.ones_like(multiplier)
    )


def _hadamard(values: torch.Tensor, block: int) -> torch.Tensor:
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


def _linear_sign_candidates() -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260819)
    random = torch.randint(0, 2, (7, 64), generator=generator).to(torch.float32)
    return torch.cat((torch.ones(1, 64), random.mul_(2).sub_(1)))


def _peak_ratio(values: torch.Tensor) -> torch.Tensor:
    peak = values.abs().amax(-1).square().sum(0)
    energy = values.square().sum(dim=(0, 2))
    return peak / energy.clamp_min(torch.finfo(torch.float32).tiny)


def _select_linear_signs(
    activation: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    group_count = activation.shape[-1] // 64
    a = activation.reshape(-1, group_count, 64)
    w = weight.reshape(-1, group_count, 64)
    candidates = _linear_sign_candidates()
    scores = [
        _peak_ratio(_hadamard(a * signs, 64))
        + _peak_ratio(_hadamard(w * signs, 64))
        for signs in candidates
    ]
    return candidates[torch.stack(scores).argmin(0)].reshape(-1)


def _linear_transform(values: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
    return _hadamard(
        values.to(torch.float32) * state["coefficient"],
        int(state["hadamard"]),
    )


# =============================================================================
# Linear: v138 output-aware calibration
# =============================================================================

@torch.inference_mode()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
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
    signs = _select_linear_signs(transformed_activation, transformed_weight)
    coefficient = multiplier * signs
    transformed_activation = _hadamard(
        transformed_activation * signs, _LINEAR_HADAMARD
    )
    transformed_weight = _hadamard(transformed_weight * signs, _LINEAR_HADAMARD)
    importance = transformed_activation.square().mean(0)
    activation_state = {
        "version": 14,
        "coefficient": coefficient,
        "hadamard": _LINEAR_HADAMARD,
    }
    baseline = _quantize_linear_values(transformed_weight, importance)
    weight_params = _output_aware_quantize_weight(
        transformed_activation,
        transformed_weight,
        baseline,
    )
    transformed_activation_parts = [
        _linear_transform(part, activation_state) for part in activation_parts
    ]
    weight_params = _refine_weight_codes(
        activation_parts,
        transformed_activation_parts,
        weight,
        weight_params,
    )
    return {"weight_params": weight_params, "activation_state": activation_state}


@torch.inference_mode()
def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    values = dequantize_nvfp4(activation_quant, activation_scale)
    return _quantize_linear_values(_linear_transform(values, activation_state))


# =============================================================================
# Attention
# =============================================================================

def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """Attention calibration state for the current version."""
    rotation = _make_qk_rotation_state(head_dim, block_size=16)
    return {
        "q_state": rotation,
        "k_state": rotation,
        "v_state": None,
    }


def _make_qk_rotation_state(
    head_dim: int,
    block_size: int = 16,
) -> dict[str, torch.Tensor]:
    """Create a shared Q/K Hadamard rotation state."""
    if head_dim % block_size != 0:
        raise ValueError("head_dim must be divisible by rotation block_size")
    h = torch.tensor([[1.0]])
    while h.shape[0] < block_size:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
        ) / math.sqrt(2.0)
    torch.manual_seed(0)
    signs = torch.randint(0, 2, (head_dim,)) * 2 - 1
    return {
        "block_size": int(block_size),
        "rot16": h.to(torch.float32),
        "signs": signs.to(torch.float32),
    }


def _apply_qk_rotation(
    value: torch.Tensor,
    num_heads: int,
    head_dim: int,
    state: dict[str, Any],
) -> torch.Tensor:
    """Apply the shared orthogonal rotation to Q or K."""
    block_size = int(state["block_size"])
    rot = state["rot16"].to(value.device, value.dtype)
    signs = state["signs"].to(value.device, value.dtype)
    seq = int(value.shape[0])
    x = value.reshape(seq, num_heads, head_dim // block_size, block_size)
    x = x @ rot
    x = x.reshape(seq, num_heads, head_dim) * signs
    return x.reshape(seq, num_heads * head_dim)


def _run_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    seq = int(q.shape[0])
    q = q.reshape(seq, q_num_heads, head_dim)
    k = k.reshape(seq, kv_num_heads, head_dim)
    v = v.reshape(seq, kv_num_heads, head_dim)
    ratio = q_num_heads // kv_num_heads
    k = k.repeat_interleave(ratio, dim=1)
    v = v.repeat_interleave(ratio, dim=1)
    scores = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
    probs = torch.softmax(scores, dim=-1)
    return probs @ v


def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    value = dequantize_nvfp4(q_quant, q_scale).to(torch.float32)
    if isinstance(q_state, dict) and "rot16" in q_state:
        value = _apply_qk_rotation(value, q_num_heads, head_dim, q_state)
    return _quantize_hif4(value, search_candidates=True, candidate_radius=1)


def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    value = dequantize_nvfp4(k_quant, k_scale).to(torch.float32)
    if isinstance(k_state, dict) and "rot16" in k_state:
        value = _apply_qk_rotation(value, kv_num_heads, head_dim, k_state)
    return _quantize_hif4(value, search_candidates=True, candidate_radius=1)


def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    value = dequantize_nvfp4(v_quant, v_scale)
    return _quantize_hif4(value, search_candidates=True, candidate_radius=1)
