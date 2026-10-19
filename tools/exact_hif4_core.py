"""Exact HiF4 encoder: per-block optimal metadata under the real format constraints.

Format constraints (HiFloat4 paper, Appendix A / Algorithm 1):
  * one E6M2 `scale_factor` per 64-element block -- shared by all 8 level-2 groups
  * one E1 bit per level-2 group (8 groups)
  * one E1 bit per level-3 group (16 groups: 2 per level-2 group, 4 elements each)
  * S1P2 mantissa in {0, 0.25, ..., 1.75}
  * dequant:  sign * mant * scale_factor * scale_lv2 * scale_lv3
    with scale_lv2 = 2^e1_8 and scale_lv3 = 2^e1_16, so the effective per-element unit is
    U = scale_factor * 2^e1_8 * 2^e1_16, and mant = round(|x/U| * 4) / 4 clamped to 1.75.

`quantize_to_hif4` fixes scale_factor = round_up(vmax/7) and thresholds the micro-exponents
at (v8/e6m2 >= 4) and (v16/(e6m2*2^e1_8) >= 2). Both are heuristics.

This encoder is exact within the constraint set. For a fixed scale_factor, the 8 level-2
groups are independent given their three bits, so each can be solved by enumerating its 8
combinations. The scale_factor itself is chosen once per block by minimising the block total.
No fitted parameters: the encoder is a deterministic function of its input block and cannot
overfit a calibration set.
"""
from __future__ import annotations

import torch

E6M2_MIN = 2.0 ** -48
_E6_SHIFTS = (-2.0, -1.0, 0.0, 1.0)
_E6_MANTS = (0.0, 1.0, 2.0, 3.0)
_COMBO_A = torch.tensor([a for a in (0, 1) for _ in (0, 1) for _ in (0, 1)])
_COMBO_B0 = torch.tensor([b0 for _ in (0, 1) for b0 in (0, 1) for _ in (0, 1)])
_COMBO_B1 = torch.tensor([b1 for _ in (0, 1) for _ in (0, 1) for b1 in (0, 1)])
_K = _COMBO_A.numel()


def e6m2_candidates(vmax: torch.Tensor) -> torch.Tensor:
    """Neighbourhood of round_up(vmax/7): (Ci, *vmax.shape)."""
    e0 = torch.floor(torch.log2((vmax / 7.0).clamp_min(1e-30)))
    out = []
    for de in _E6_SHIFTS:
        for m in _E6_MANTS:
            out.append(((2.0 ** (e0 + de)) * (1.0 + m / 4.0)).clamp_min(E6M2_MIN))
    return torch.stack(out, 0)


@torch.no_grad()
def exact_hif4_params(x2d: torch.Tensor, chunk_rows: int = 256) -> dict[str, torch.Tensor]:
    """x2d: (R, C) fp32, C % 64 == 0 -> HiF4 params in the quantize_to_hif4 layout."""
    x2d = x2d.float()
    R, C = x2d.shape
    assert C % 64 == 0, C
    b = C // 64
    lv2_k = (2.0 ** _COMBO_A.float()).view(_K, 1, 1, 1, 1)
    lv3_k = torch.stack([2.0 ** _COMBO_B0.float(), 2.0 ** _COMBO_B1.float()], -1).view(
        _K, 1, 1, 1, 2)
    a_of = _COMBO_A.float()
    b0_of = _COMBO_B0.float()
    b1_of = _COMBO_B1.float()
    sf_l, lv2_l, lv3_l, sign_l, mant_l = [], [], [], [], []
    for s in range(0, R, chunk_rows):
        xc = x2d[s:s + chunk_rows]
        n = xc.shape[0]
        g = xc.reshape(n, b, 8, 2, 4)
        v16 = g.abs().amax(-1)                     # (n,b,8,2)
        v8 = v16.amax(-1)                          # (n,b,8)
        vmax = v8.amax(-1)                         # (n,b)
        cand = e6m2_candidates(vmax)               # (Ci,n,b)
        ci_n = cand.shape[0]
        err_all = torch.empty(ci_n, _K, n, b, 8)
        for ci in range(ci_n):
            unit = (cand[ci].view(1, n, b, 1, 1, 1) * lv2_k.view(_K, 1, 1, 1, 1, 1)
                    * lv3_k.view(_K, 1, 1, 1, 2, 1))          # (K,n,b,8,2,1)
            vs = g.unsqueeze(0) / unit
            mant = (torch.round(vs.abs() * 4.0) / 4.0).clamp(0.0, 1.75)
            deq = torch.sign(vs) * mant * unit
            err_all[ci] = (deq - g.unsqueeze(0)).square().sum((-1, -2))
            del unit, vs, mant, deq
        per_group_best = err_all.amin(1)           # (Ci,n,b,8)
        block_total = per_group_best.sum(-1)       # (Ci,n,b)
        ci_star = block_total.argmin(0)            # (n,b)
        chosen = err_all.permute(2, 3, 0, 1, 4).reshape(n * b, ci_n, _K, 8)[
            torch.arange(n * b), ci_star.reshape(-1)].reshape(n, b, _K, 8)
        combo_star = chosen.argmin(2)              # (n,b,8)
        ar = torch.arange(n * b)
        sf = cand.permute(1, 2, 0).reshape(n * b, ci_n)[ar, ci_star.reshape(-1)].reshape(n, b)
        aa = a_of[combo_star.reshape(-1)].reshape(n, b, 8)
        bb0 = b0_of[combo_star.reshape(-1)].reshape(n, b, 8)
        bb1 = b1_of[combo_star.reshape(-1)].reshape(n, b, 8)
        lv2 = 2.0 ** aa                            # (n,b,8)
        lv3 = torch.stack([2.0 ** bb0, 2.0 ** bb1], -1)   # (n,b,8,2)
        unit = sf.view(n, b, 1, 1, 1) * lv2.view(n, b, 8, 1, 1) * lv3.view(n, b, 8, 2, 1)
        vs = g / unit
        mant = (torch.round(vs.abs() * 4.0) / 4.0).clamp(0.0, 1.75)
        sf_l.append(sf.reshape(n, b, 1, 1, 1))
        lv2_l.append(lv2.reshape(n, b, 8, 1, 1))
        lv3_l.append(lv3.reshape(n, b, 8, 2, 1))
        sign_l.append(torch.sign(vs).reshape(n, b, 8, 2, 4))
        mant_l.append(mant.reshape(n, b, 8, 2, 4))
    return {'scale_factor': torch.cat(sf_l, 0), 'scale_lv2': torch.cat(lv2_l, 0),
            'scale_lv3': torch.cat(lv3_l, 0), 'sign': torch.cat(sign_l, 0),
            'mant': torch.cat(mant_l, 0)}


@torch.no_grad()
def exact_hif4_dequant(x2d: torch.Tensor, chunk_rows: int = 256) -> torch.Tensor:
    p = exact_hif4_params(x2d, chunk_rows)
    return (p['sign'] * p['mant'] * p['scale_lv2'] * p['scale_lv3']
            * p['scale_factor']).reshape(x2d.shape)
