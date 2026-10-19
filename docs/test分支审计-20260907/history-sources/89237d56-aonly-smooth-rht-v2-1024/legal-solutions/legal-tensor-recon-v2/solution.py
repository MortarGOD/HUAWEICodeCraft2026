"""Legal HiF4 baseline using only independent tensor reconstruction objectives."""

from __future__ import annotations

import math
from typing import Any

import torch


_MIN_SCALE_FACTOR = 2.0**-48
_MAX_SCALE_FACTOR = 49152.0
_MANTISSA_MAX = 1.75
_MANTISSA_STEP = 0.25
_LINEAR_SCALE_RANK_OFFSETS = (-3, -2, -1, 0, 1, 2, 3)
_ATTENTION_QK_SCALE_RANK_OFFSETS = (-3, -2, -1, 0, 1, 2, 3)
_ATTENTION_V_SCALE_RANK_OFFSETS = (-3, -2, -1, 0, 1, 2, 3)
_LINEAR_SMOOTH_ALPHA = 0.388671875
_LINEAR_SMOOTH_QUANTILE = 0.0
_LINEAR_SMOOTH_POWER = 2.0
_LINEAR_SMOOTH_LOG2_LIMIT = 3.0
_LINEAR_BALANCE_CENTER_BLOCK = 1024
_LINEAR_SIGN_SEED = 0
_ATTENTION_ROTATION_BLOCK = 64
_ATTENTION_BALANCE_EXPONENT = 0.359375
_ATTENTION_BALANCE_LOG2_LIMIT = 2.0
_ATTENTION_SIGN_SEED = 0


def _dequantize_src(quant: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if not isinstance(quant, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise TypeError("quant and scale must be tensors")
    if quant.ndim < 1 or quant.shape[-1] % 16 != 0:
        raise ValueError("quant last dimension must be divisible by 16")
    expected_scale_shape = quant.shape[:-1] + (quant.shape[-1] // 16,)
    if tuple(scale.shape) != tuple(expected_scale_shape):
        raise ValueError("scale shape does not match quant")
    value = (
        quant.detach()
        .to(device="cpu")
        .unflatten(-1, (-1, 16))
        .to(torch.float32)
        * scale.detach().to(device="cpu", dtype=torch.float32).unsqueeze(-1)
    )
    value = value.flatten(-2, -1).to(torch.bfloat16).to(torch.float32)
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError("source tensor contains NaN or infinity")
    return value.contiguous()


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _fixed_signs(size: int, seed: int) -> torch.Tensor:
    if seed < 0:
        return torch.ones(size, dtype=torch.float32)
    indices = torch.arange(size, dtype=torch.int64)
    indices = indices + seed * size
    hashed = indices * 6364136223846793005 + 1442695040888963407
    return torch.where(
        torch.bitwise_and(torch.bitwise_right_shift(hashed, 32), 1) == 0,
        1.0,
        -1.0,
    ).to(torch.float32)


def _signed_fwht_last_dim(value: torch.Tensor, seed: int) -> torch.Tensor:
    size = int(value.shape[-1])
    if not _is_power_of_two(size):
        return value
    transformed = value.to(torch.float32) * _fixed_signs(size, seed)
    width = 1
    while width < size:
        paired = transformed.reshape(transformed.shape[:-1] + (-1, 2, width))
        left = paired[..., 0, :]
        right = paired[..., 1, :]
        transformed = torch.cat((left + right, left - right), dim=-1).reshape(
            value.shape
        )
        width *= 2
    return transformed * (size**-0.5)


def _rotate_linear(value: torch.Tensor) -> torch.Tensor:
    shape = tuple(value.shape)
    blocked = value.reshape(shape[:-1] + (shape[-1] // 64, 64))
    return _signed_fwht_last_dim(blocked, _LINEAR_SIGN_SEED).reshape(shape).contiguous()


def _rotate_attention_heads(
    value: torch.Tensor, num_heads: int, head_dim: int
) -> torch.Tensor:
    shape = tuple(value.shape)
    headed = value.reshape(shape[:-1] + (num_heads, head_dim))
    if _ATTENTION_ROTATION_BLOCK == 0:
        rotated = _signed_fwht_last_dim(headed, _ATTENTION_SIGN_SEED)
    else:
        block = _ATTENTION_ROTATION_BLOCK
        if head_dim % block != 0 or not _is_power_of_two(block):
            raise ValueError("invalid attention rotation block")
        blocked = headed.reshape(shape[:-1] + (num_heads, head_dim // block, block))
        rotated = _signed_fwht_last_dim(
            blocked, _ATTENTION_SIGN_SEED
        ).reshape(headed.shape)
    return rotated.reshape(shape).contiguous()


def _e6m2_candidates(
    block_peak: torch.Tensor, scale_rank_offsets: tuple[int, ...]
) -> torch.Tensor:
    target = torch.clamp(
        block_peak.to(torch.float64) / (4.0 * _MANTISSA_MAX),
        min=_MIN_SCALE_FACTOR,
        max=_MAX_SCALE_FACTOR,
    )
    exponent = torch.floor(torch.log2(target))
    significand_index = torch.ceil(target / torch.pow(2.0, exponent) * 4.0) - 4.0
    carry = torch.div(significand_index.to(torch.int64), 4, rounding_mode="floor")
    base_rank = (exponent.to(torch.int64) + carry) * 4 + torch.remainder(
        significand_index.to(torch.int64), 4
    )
    offsets = torch.tensor(
        scale_rank_offsets, dtype=torch.int64, device=block_peak.device
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


def _quantize_at_scale(
    x: torch.Tensor,
    absolute: torch.Tensor,
    scale_factor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    best_group_error: torch.Tensor | None = None
    best_lv2: torch.Tensor | None = None
    best_lv3: torch.Tensor | None = None
    best_mantissa: torch.Tensor | None = None

    for lv2_value in (1.0, 2.0):
        quartet_errors: list[torch.Tensor] = []
        quartet_mantissas: list[torch.Tensor] = []
        for lv3_value in (1.0, 2.0):
            effective_scale = scale_factor * (lv2_value * lv3_value)
            mantissa = torch.clamp(
                torch.round(absolute / effective_scale / _MANTISSA_STEP)
                * _MANTISSA_STEP,
                min=0.0,
                max=_MANTISSA_MAX,
            )
            reconstructed = torch.sign(x) * mantissa * effective_scale
            quartet_errors.append(
                torch.sum((reconstructed - x).square(), dim=-1, keepdim=True)
            )
            quartet_mantissas.append(mantissa)

        use_high_lv3 = quartet_errors[1] < quartet_errors[0]
        lv3 = torch.where(use_high_lv3, 2.0, 1.0).to(torch.float32)
        mantissa = torch.where(
            use_high_lv3, quartet_mantissas[1], quartet_mantissas[0]
        )
        quartet_error = torch.where(
            use_high_lv3, quartet_errors[1], quartet_errors[0]
        )
        group_error = torch.sum(quartet_error, dim=-2, keepdim=True)

        if best_group_error is None:
            best_group_error = group_error
            best_lv2 = torch.full_like(group_error, lv2_value)
            best_lv3 = lv3
            best_mantissa = mantissa
        else:
            use_this_lv2 = group_error < best_group_error
            best_group_error = torch.where(
                use_this_lv2, group_error, best_group_error
            )
            best_lv2 = torch.where(
                use_this_lv2, torch.full_like(best_lv2, lv2_value), best_lv2
            )
            best_lv3 = torch.where(use_this_lv2, lv3, best_lv3)
            best_mantissa = torch.where(use_this_lv2, mantissa, best_mantissa)

    assert best_group_error is not None
    assert best_lv2 is not None
    assert best_lv3 is not None
    assert best_mantissa is not None
    block_error = torch.sum(best_group_error, dim=(-3, -2, -1), keepdim=True)
    sign = torch.where(best_mantissa == 0.0, 0.0, torch.sign(x))
    return block_error, best_lv2, best_lv3, sign, best_mantissa


def _quantize_tensor(
    value: torch.Tensor,
    scale_rank_offsets: tuple[int, ...] = _LINEAR_SCALE_RANK_OFFSETS,
) -> dict[str, torch.Tensor]:
    if not isinstance(value, torch.Tensor) or value.ndim < 1:
        raise TypeError("value must be a non-scalar tensor")
    channels = int(value.shape[-1])
    if channels % 64 != 0:
        raise ValueError("value last dimension must be divisible by 64")

    original_shape = tuple(value.shape)
    blocked_shape = original_shape[:-1] + (channels // 64, 8, 2, 4)
    x = value.detach().to(device="cpu", dtype=torch.float32).contiguous().reshape(
        blocked_shape
    )
    if not bool(torch.isfinite(x).all().item()):
        raise ValueError("value contains NaN or infinity")

    absolute = torch.abs(x)
    block_peak = torch.amax(absolute, dim=(-3, -2, -1), keepdim=True)
    candidates = _e6m2_candidates(block_peak, scale_rank_offsets)

    best_error: torch.Tensor | None = None
    best_scale: torch.Tensor | None = None
    best_lv2: torch.Tensor | None = None
    best_lv3: torch.Tensor | None = None
    best_sign: torch.Tensor | None = None
    best_mantissa: torch.Tensor | None = None
    for candidate_index in range(candidates.shape[-1]):
        scale_factor = candidates[..., candidate_index]
        error, lv2, lv3, sign, mantissa = _quantize_at_scale(
            x, absolute, scale_factor
        )
        if best_error is None:
            best_error = error
            best_scale = scale_factor
            best_lv2 = lv2
            best_lv3 = lv3
            best_sign = sign
            best_mantissa = mantissa
        else:
            use_candidate = error < best_error
            best_error = torch.where(use_candidate, error, best_error)
            best_scale = torch.where(use_candidate, scale_factor, best_scale)
            best_lv2 = torch.where(use_candidate, lv2, best_lv2)
            best_lv3 = torch.where(use_candidate, lv3, best_lv3)
            best_sign = torch.where(use_candidate, sign, best_sign)
            best_mantissa = torch.where(use_candidate, mantissa, best_mantissa)

    assert best_scale is not None
    assert best_lv2 is not None
    assert best_lv3 is not None
    assert best_sign is not None
    assert best_mantissa is not None
    return {
        "scale_factor": best_scale.contiguous(),
        "scale_lv2": best_lv2.contiguous(),
        "scale_lv3": best_lv3.contiguous(),
        "sign": best_sign.contiguous(),
        "mant": best_mantissa.contiguous(),
    }


def _dequantize_tgt(
    params: dict[str, torch.Tensor], original_shape: tuple[int, ...]
) -> torch.Tensor:
    reconstructed = (
        params["sign"].to(torch.float32)
        * params["mant"].to(torch.float32)
        * params["scale_factor"].to(torch.float32)
        * params["scale_lv2"].to(torch.float32)
        * params["scale_lv3"].to(torch.float32)
    )
    return reconstructed.reshape(original_shape)


def _linear_activation_balance(
    calib_activation_list: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    moment: torch.Tensor | None = None
    absolute_samples: list[torch.Tensor] = []
    rows = 0
    for quant, scale in calib_activation_list:
        activation = _dequantize_src(quant, scale)
        sample_moment = torch.sum(
            torch.abs(activation).pow(_LINEAR_SMOOTH_POWER), dim=0
        )
        moment = sample_moment if moment is None else moment + sample_moment
        if _LINEAR_SMOOTH_QUANTILE > 0.0:
            absolute_samples.append(torch.abs(activation))
        rows += int(activation.shape[0])
    assert moment is not None
    if _LINEAR_SMOOTH_QUANTILE > 0.0:
        statistic = torch.quantile(
            torch.cat(absolute_samples, dim=0),
            _LINEAR_SMOOTH_QUANTILE,
            dim=0,
        )
    else:
        statistic = torch.pow(
            moment / max(rows, 1), 1.0 / _LINEAR_SMOOTH_POWER
        )
    statistic = statistic.clamp_min(1.0e-24)
    center_block = (
        int(statistic.numel())
        if _LINEAR_BALANCE_CENTER_BLOCK == 0
        else math.gcd(int(statistic.numel()), _LINEAR_BALANCE_CENTER_BLOCK)
    )
    log_statistic = torch.log2(statistic).reshape(-1, center_block)
    log_statistic = log_statistic - torch.mean(
        log_statistic, dim=-1, keepdim=True
    )
    log_balance = torch.clamp(
        -_LINEAR_SMOOTH_ALPHA * log_statistic,
        min=-_LINEAR_SMOOTH_LOG2_LIMIT,
        max=_LINEAR_SMOOTH_LOG2_LIMIT,
    )
    return torch.pow(2.0, log_balance).flatten().contiguous()


def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, Any]:
    if type(calib_activation_list) is not list or not calib_activation_list:
        raise ValueError("calib_activation_list must be a non-empty list")
    balance = _linear_activation_balance(calib_activation_list)
    weight = _dequantize_src(weight_quant, weight_scale)
    return {
        "weight_params": _quantize_tensor(_rotate_linear(weight / balance)),
        "activation_state": {"balance": balance},
    }


def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    if type(activation_state) is not dict or set(activation_state) != {"balance"}:
        raise ValueError("activation_state must contain only balance")
    balance = activation_state["balance"]
    if not isinstance(balance, torch.Tensor) or balance.ndim != 1:
        raise ValueError("activation balance must be a rank-1 tensor")
    activation = _dequantize_src(activation_quant, activation_scale)
    if balance.shape[0] != activation.shape[-1]:
        raise ValueError("activation balance width mismatch")
    return _quantize_tensor(_rotate_linear(activation * balance))


def hif4_calibration_attention(
    calib_qkv_list: list[dict[str, tuple[torch.Tensor, torch.Tensor]]],
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    if type(calib_qkv_list) is not list or not calib_qkv_list:
        raise ValueError("calib_qkv_list must be a non-empty list")
    if type(q_num_heads) is not int or q_num_heads <= 0:
        raise ValueError("q_num_heads must be a positive int")
    if type(kv_num_heads) is not int or kv_num_heads <= 0:
        raise ValueError("kv_num_heads must be a positive int")
    if type(head_dim) is not int or head_dim <= 0:
        raise ValueError("head_dim must be a positive int")
    if q_num_heads % kv_num_heads != 0:
        raise ValueError("q_num_heads must be divisible by kv_num_heads")
    q_energy = torch.zeros((kv_num_heads, head_dim), dtype=torch.float32)
    k_energy = torch.zeros_like(q_energy)
    q_rows = 0
    k_rows = 0
    q_per_kv = q_num_heads // kv_num_heads
    for sample in calib_qkv_list:
        q = _dequantize_src(*sample["q"]).reshape(
            -1, kv_num_heads, q_per_kv, head_dim
        )
        k = _dequantize_src(*sample["k"]).reshape(
            -1, kv_num_heads, head_dim
        )
        q_energy += torch.sum(q.square(), dim=(0, 2))
        k_energy += torch.sum(k.square(), dim=0)
        q_rows += int(q.shape[0]) * q_per_kv
        k_rows += int(k.shape[0])
    q_stat = torch.sqrt(q_energy / max(q_rows, 1)).clamp_min(1.0e-24)
    k_stat = torch.sqrt(k_energy / max(k_rows, 1)).clamp_min(1.0e-24)
    log_balance = _ATTENTION_BALANCE_EXPONENT * (
        torch.log2(k_stat) - torch.log2(q_stat)
    )
    log_balance = log_balance.reshape(kv_num_heads, -1, 64)
    log_balance = log_balance - torch.mean(
        log_balance, dim=-1, keepdim=True
    )
    log_balance = torch.clamp(
        log_balance,
        min=-_ATTENTION_BALANCE_LOG2_LIMIT,
        max=_ATTENTION_BALANCE_LOG2_LIMIT,
    )
    balance = torch.pow(2.0, log_balance).reshape(kv_num_heads, head_dim)
    state = {"balance": balance.contiguous()}
    return {"q_state": state, "k_state": state, "v_state": None}


def _dynamic_attention(
    quant: torch.Tensor,
    scale: torch.Tensor,
    num_heads: int,
    head_dim: int,
    state: Any,
    stream: str,
    rotate: bool,
) -> dict[str, torch.Tensor]:
    if type(num_heads) is not int or num_heads <= 0:
        raise ValueError("num_heads must be a positive int")
    if type(head_dim) is not int or head_dim <= 0:
        raise ValueError("head_dim must be a positive int")
    if quant.ndim != 2 or quant.shape[-1] != num_heads * head_dim:
        raise ValueError(f"{stream} width does not match num_heads * head_dim")
    value = _dequantize_src(quant, scale)
    if stream in {"q", "k"}:
        if type(state) is not dict or set(state) != {"balance"}:
            raise ValueError(f"invalid {stream}_state")
        balance = state["balance"]
        if not isinstance(balance, torch.Tensor) or balance.ndim != 2:
            raise ValueError(f"invalid {stream} balance")
        kv_heads = int(balance.shape[0])
        if balance.shape[1] != head_dim or num_heads % kv_heads != 0:
            raise ValueError(f"invalid {stream} balance shape")
        headed = value.reshape(-1, num_heads, head_dim)
        head_balance = balance.repeat_interleave(num_heads // kv_heads, dim=0)
        if stream == "q":
            headed = headed * head_balance.unsqueeze(0)
        else:
            headed = headed / head_balance.unsqueeze(0)
        value = headed.reshape(value.shape)
    elif state is not None:
        raise ValueError(f"{stream}_state must be None")
    if rotate:
        value = _rotate_attention_heads(value, num_heads, head_dim)
    offsets = (
        _ATTENTION_QK_SCALE_RANK_OFFSETS
        if stream in {"q", "k"}
        else _ATTENTION_V_SCALE_RANK_OFFSETS
    )
    return _quantize_tensor(value, offsets)


def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    return _dynamic_attention(
        q_quant, q_scale, q_num_heads, head_dim, q_state, "q", True
    )


def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    return _dynamic_attention(
        k_quant, k_scale, kv_num_heads, head_dim, k_state, "k", True
    )


def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    return _dynamic_attention(
        v_quant, v_scale, kv_num_heads, head_dim, v_state, "v", False
    )


calibrate_and_quantize_weight = hif4_calibration_and_quantize_weight
dynamic_quantize_activation = hif4_dynamic_quantize_activation
calibrate_attention = hif4_calibration_attention
dynamic_quantize_q = hif4_dynamic_quantize_q
dynamic_quantize_k = hif4_dynamic_quantize_k
dynamic_quantize_v = hif4_dynamic_quantize_v
schema_version = 1
