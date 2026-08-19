"""
NVFP4 -> HiF4 baseline converter for the 2026 Huawei Algorithm Contest.

The implementation first decodes the incoming NVFP4 carrier/block-scale
representation back to BF16 using the public ``dequantize_nvfp4`` helper, and
then quantizes the BF16 tensor into the HiF4 block floating-point format using
the direct-cast conversion described in the HiFloat4 paper (Algorithm 1).
"""

from __future__ import annotations

from typing import Any

import torch


E6M2_MIN = 2.0 ** (-48)
E6M2_MAX = 49152.0  # 2^15 * 1.5


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


def _quantize_hif4(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Convert a BF16/FP32 tensor to HiF4 parameters along its last dim.

    The last dimension must be divisible by the HiF4 group size 64.
    """
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

    # Stage 1: hierarchical peak reduction.
    v16 = xb.abs().amax(dim=-1)          # [*prefix, nblocks, 8, 2]
    v8 = v16.amax(dim=-1)                # [*prefix, nblocks, 8]
    vmax = v8.amax(dim=-1, keepdim=True)  # [*prefix, nblocks, 1]

    # Stage 2: three-level scaling metadata.
    sf_bf16 = vmax.squeeze(-1) / 7.0
    e6m2 = _e6m2_quant(sf_bf16)          # [*prefix, nblocks]
    e6m2_rec = 1.0 / e6m2

    e1_8 = (v8 * e6m2_rec.unsqueeze(-1) >= 4.0).to(torch.float32)
    e1_8_pairs = e1_8.unsqueeze(-1)
    e1_16 = (
        v16
        * e6m2_rec.unsqueeze(-1).unsqueeze(-1)
        * (2.0 ** (-e1_8_pairs))
        >= 2.0
    ).to(torch.float32)

    # Stage 3: element-wise S1P2 quantization.
    rec_for = e6m2_rec.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    e1_8_for = e1_8.unsqueeze(-1).unsqueeze(-1)
    e1_16_for = e1_16.unsqueeze(-1)
    scaled = xb * rec_for * (2.0 ** (-e1_8_for)) * (2.0 ** (-e1_16_for))
    sign, mant = _s1p2_quant(scaled)

    return {
        "scale_factor": e6m2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
        "scale_lv2": (2.0 ** e1_8).unsqueeze(-1).unsqueeze(-1),
        "scale_lv3": (2.0 ** e1_16).unsqueeze(-1),
        "sign": sign,
        "mant": mant,
    }


# =============================================================================
# Linear
# =============================================================================

def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    """Decode NVFP4 weight and convert it to HiF4 weight parameters."""
    weight = dequantize_nvfp4(weight_quant, weight_scale)
    weight_params = _quantize_hif4(weight)
    return {
        "weight_params": weight_params,
        "activation_state": None,
    }


def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    """Decode NVFP4 activation and convert it to HiF4 parameters."""
    activation = dequantize_nvfp4(activation_quant, activation_scale)
    return _quantize_hif4(activation)


# =============================================================================
# Attention
# =============================================================================

def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """Return empty calibration states for the direct-cast baseline."""
    return {
        "q_state": None,
        "k_state": None,
        "v_state": None,
    }


def _quantize_qkv_tensor(quant: torch.Tensor, scale: torch.Tensor) -> dict[str, torch.Tensor]:
    value = dequantize_nvfp4(quant, scale)
    return _quantize_hif4(value)


def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    return _quantize_qkv_tensor(q_quant, q_scale)


def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    return _quantize_qkv_tensor(k_quant, k_scale)


def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    return _quantize_qkv_tensor(v_quant, v_scale)
