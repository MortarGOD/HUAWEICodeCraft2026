"""v643 core: encode V as the identity-optimal HiF4 of the exact tensor, over a wider scale grid.

v642 moves V's codes to the identity optimum but keeps whatever scale factors the parent chain
produced. Re-picking the scales inside the parent's 6-candidate grid turns out to be a wash
(online_v1 0.678008 -> 0.677260, online_profiled_v2 0.678131 -> 0.676592), but WIDENING that grid
is not: 40 candidates give 0.694902 and 0.693550, i.e. +422 and +385 on the two peaked banks.

_quantize_weighted already solves the whole per-block problem exactly inside a grid - for each
scale candidate it picks the lv tree optimally and rounds the mantissa - so passing a wider grid
yields the identity optimum over that larger family, with no fitting, no counterpart and no state.
The parent's params are kept as the fallback if anything goes wrong.

The extra cost is the grid width; the grid is a structure-agnostic constant, not a fitted
quantity, so it is not bank-specific the way the surrogate metric was.
"""
import torch

# offsets -2..2 crossed with 3-bit mantissa multipliers 1.0..1.875 (1/8 steps)
_V643_GRID = tuple((off, sig / 8.0)
                   for off in (-2, -1, 0, 1, 2)
                   for sig in range(8, 16))

_v643_parent_v = hif4_dynamic_quantize_v


@torch.inference_mode()
def hif4_dynamic_quantize_v(vq, vs, kh, d, state):
    params = _v643_parent_v(vq, vs, kh, d, state)
    try:
        target = dequantize_nvfp4(vq, vs).float()
        ones = torch.ones(1, target.shape[1], dtype=torch.float32)
        return _quantize_weighted(target, ones, grid=_V643_GRID)
    except (RuntimeError, ValueError, TypeError, KeyError):
        return params
