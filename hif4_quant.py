#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standard HiF4 quantizer, following the HiFloat4 paper (Algorithm 1).

Format recap (HiFloat4 Format for Language Model Inference):
  - A HiF4 unit = 64 4-bit S1P2 elements + 32-bit shared scaling metadata:
      * E6M2: unsigned 8-bit float, 6-bit exponent (bias 48) + 2-bit mantissa
              with hidden integer bit 1. X = 2^E * 1.M, E in [-48, 15].
              Range [2^-48, 49152]; NaN = 111111_11.
      * E1_8:  8 x 1-bit micro-exponents, value {0, 1}
      * E1_16: 16 x 1-bit micro-exponents, value {0, 1}
      * S1P2:  sign-magnitude, values {0, +-0.25, ..., +-1.75}
  - Dequant:  V_i = E6M2 * 2^(E1_8[ceil(i/8)] + E1_16[ceil(i/4)]) * S1P2_i
  - Quant (Algorithm 1): 3-level peak reduction, SF = Vmax/7, E6M2 scale,
    micro-exponent compare (>=4, >=2), scale, then S1P2 rounding.
"""

from __future__ import annotations

from typing import Any

import torch

E6M2_MIN = 2.0 ** -48
E6M2_MAX = 49152.0  # 2^15 * 1.50


def dequantize_nvfp4(quant_float: torch.Tensor,
                     scale_float: torch.Tensor,
                     blk_size: int = 16) -> torch.Tensor:
    """NVFP4 dequantization (per competition API doc)."""
    c = quant_float.shape[-1]
    assert c % blk_size == 0, f"Last dim {c} not divisible by NVFP4 block size {blk_size}"
    x = quant_float.unflatten(-1, (-1, blk_size))
    sf = scale_float.unsqueeze(-1)
    result = x * sf
    result = result.flatten(-2, -1)
    return result.to(torch.bfloat16)


def _quantize_e6m2_up(sf: torch.Tensor) -> torch.Tensor:
    """Round positive floats UP to the nearest representable E6M2 value.

    E6M2 represents X = 2^E * (1 + m/4) with E in [-48, 15], m in {0,1,2,3}.
    Rounding up (rather than to-nearest) avoids saturating the S1P2 elements
    (same convention as NVFP4 scale rounding, NVFP4 paper Appendix B.4).
    """
    sf = torch.where(sf <= 0.0, torch.full_like(sf, E6M2_MIN), sf)
    e = torch.floor(torch.log2(sf))
    man = sf / (2.0 ** e)            # in [1, 2)
    m = torch.ceil((man - 1.0) * 4.0)  # in {0,1,2,3,4}
    carry = m >= 4.0
    m = torch.where(carry, torch.zeros_like(m), m)
    e = torch.where(carry, e + 1.0, e)
    e = torch.clamp(e, -48.0, 15.0)
    val = (2.0 ** e) * (1.0 + m / 4.0)
    return val.clamp(E6M2_MIN, E6M2_MAX)


def quantize_to_hif4(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Convert a high-precision tensor to HiF4 parameters (Algorithm 1).

    Args:
        x: FP32/BF16 tensor, last dim divisible by 64.
    Returns:
        dict with keys scale_factor, scale_lv2, scale_lv3, sign, mant,
        shaped per the competition HiF4Params layout.
    """
    x = x.float()
    orig_shape = tuple(x.shape)
    c = orig_shape[-1]
    assert c % 64 == 0, f"Last dim {c} not divisible by HiF4 block size 64"
    b = c // 64

    x64 = x.unflatten(-1, (-1, 64))                 # [..., B, 64]
    g = x64.reshape(*x64.shape[:-1], 8, 2, 4)       # [..., B, 8, 2, 4]

    # ---- Stage 1: three-level peak reduction (Algorithm 1 lines 1-7) ----
    v16 = g.abs().amax(dim=-1)                       # [..., B, 8, 2]
    v8 = v16.amax(dim=-1)                            # [..., B, 8]
    vmax = v8.amax(dim=-1)                           # [..., B]

    # ---- Stage 2: hierarchical scaling metadata (lines 8-14) ----
    sf = vmax * (1.0 / 7.0)                          # 7 = max intra-group value
    e6m2 = _quantize_e6m2_up(sf)                     # [..., B]
    e6m2_rec = 1.0 / e6m2                            # [..., B]

    e1_8 = (v8 * e6m2_rec.unsqueeze(-1) >= 4.0).to(x.dtype)            # [..., B, 8]
    e1_8_for_v16 = e1_8.unsqueeze(-1)                                  # [..., B, 8, 1]
    cond16 = v16 * e6m2_rec.unsqueeze(-1).unsqueeze(-1) * (2.0 ** -e1_8_for_v16)
    e1_16 = (cond16 >= 2.0).to(x.dtype)                                # [..., B, 8, 2]

    # ---- Stage 3: scaled elements + S1P2 (lines 15-18) ----
    rec = e6m2_rec.reshape(*orig_shape[:-1], b, 1, 1, 1)
    lv2_rec = (2.0 ** -e1_8).reshape(*orig_shape[:-1], b, 8, 1, 1)
    lv3_rec = (2.0 ** -e1_16).reshape(*orig_shape[:-1], b, 8, 2, 1)
    v_scaled = g * rec * lv2_rec * lv3_rec          # [..., B, 8, 2, 4]

    sign = torch.sign(v_scaled)                      # {-1, 0, 1}
    mant = (torch.round(v_scaled.abs() * 4.0) / 4.0).clamp(0.0, 1.75)

    return {
        "scale_factor": e6m2.reshape(*orig_shape[:-1], b, 1, 1, 1),
        "scale_lv2": (2.0 ** e1_8).reshape(*orig_shape[:-1], b, 8, 1, 1),
        "scale_lv3": (2.0 ** e1_16).reshape(*orig_shape[:-1], b, 8, 2, 1),
        "sign": sign.reshape(*orig_shape[:-1], b, 8, 2, 4),
        "mant": mant.reshape(*orig_shape[:-1], b, 8, 2, 4),
    }


def dequantize_hif4(params: dict[str, torch.Tensor],
                    original_shape: tuple[int, ...]) -> torch.Tensor:
    """Dequantize HiF4 params back to a high-precision tensor."""
    dequant = (params["sign"] * params["mant"]
               * params["scale_lv2"] * params["scale_lv3"] * params["scale_factor"])
    return dequant.reshape(original_shape)
