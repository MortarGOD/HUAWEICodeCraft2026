"""v646 core: encode V as the identity-optimal HiF4 of the exact tensor over a wider LEGAL grid.

v643 widened the scale grid but used multipliers like 1.125, which are not representable in E6M2;
self_check rejected it 17/22 (scale_factor must be an exact multiple of 2^(sf_exp-2), i.e. 1.0,
1.25, 1.5 or 1.75), so its apparent gain came from leaving the legal parameter set. This keeps the
multipliers legal and widens only the EXPONENT offset, which is what the dynamic path's 6-candidate
CAND_GRID restricts: it uses offsets {0,1} x {1,1.25,1.5,1.75} and is even missing (1,1.5) and
(1,1.75). Here offsets (-4, -3, -2, -1, 0, 1, 2, 3) x the four legal multipliers = 32 candidates.

_quantize_weighted solves the whole per-block problem exactly inside a grid - for each scale
candidate it picks the lv tree optimally and rounds the mantissa - so a wider legal grid yields the
identity optimum over that larger family with no fitting, no counterpart and no state.
"""
import torch

_V646_GRID = tuple((off, sig) for off in (-4, -3, -2, -1, 0, 1, 2, 3)
                  for sig in (1.0, 1.25, 1.5, 1.75))

_v646_parent_v = hif4_dynamic_quantize_v


@torch.inference_mode()
def hif4_dynamic_quantize_v(vq, vs, kh, d, state):
    params = _v646_parent_v(vq, vs, kh, d, state)
    try:
        target = dequantize_nvfp4(vq, vs).float()
        ones = torch.ones(1, target.shape[1], dtype=torch.float32)
        return _quantize_weighted(target, ones, grid=_V646_GRID)
    except (RuntimeError, ValueError, TypeError, KeyError):
        return params
