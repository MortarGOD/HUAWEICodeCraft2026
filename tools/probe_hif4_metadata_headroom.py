"""How much error does HiF4's Algorithm-1 threshold heuristic leave on the table?

`quantize_to_hif4` picks the level-2/level-3 micro-exponents by fixed thresholds
(v8/e6m2 >= 4, then v16/(e6m2 * 2^e1_8) >= 2) and E6M2 as round_up(vmax/7).
Those are heuristics. Given a fixed E6M2 the eight (e1_8, e1_16[0], e1_16[1]) choices of
one level-2 group are independent of every other level-2 group, so the error-minimising
choice is found exactly by enumeration: 8 combinations per level-2 group plus a short
sweep over E6M2 candidates.

This is a measurement, not a mechanism: it says whether the quantisation operator itself
still has headroom on the bank we are fitting.

Usage: probe_hif4_metadata_headroom.py [roles] [token_stride]
"""
import sys
from pathlib import Path

import torch

ROOT = Path('/Users/ee/Code/2026-huawei-alogorithm')
sys.path.insert(0, str(ROOT))
from hif4_quant import quantize_to_hif4, dequantize_hif4, dequantize_nvfp4  # noqa: E402

torch.set_num_threads(4)
ROLES = sys.argv[1] if len(sys.argv) > 1 else 'v'
TOKEN_STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 1
COMBOS = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
E6_SHIFTS = (-2.0, -1.0, 0.0, 1.0)
E6_MANTS = (0.0, 1.0, 2.0, 3.0)


def e6m2_candidates(vmax):
    """round_up(vmax/7) neighbourhood: (C, N, B)."""
    e0 = torch.floor(torch.log2((vmax / 7.0).clamp_min(1e-30)))
    out = []
    for de in E6_SHIFTS:
        for m in E6_MANTS:
            out.append(((2.0 ** (e0 + de)) * (1.0 + m / 4.0)).clamp_min(2.0 ** -48))
    return torch.stack(out, 0)


def exact_block(x64):
    """x64: (N, B, 64) -> dequantised (N, B, 64) under per-group optimal metadata."""
    n, b = x64.shape[0], x64.shape[1]
    g = x64.reshape(n, b, 8, 2, 4)
    v16 = g.abs().amax(-1)                     # (N,B,8,2)
    v8 = v16.amax(-1)                          # (N,B,8)
    vmax = v8.amax(-1)                         # (N,B)
    cand = e6m2_candidates(vmax)               # (C,N,B)
    # scale-down factors for forming v_scaled, and the matching dequant factors
    s3_dn = torch.tensor([[2.0 ** -b0, 2.0 ** -b1] for (_, b0, b1) in COMBOS])
    s3_up = torch.tensor([[2.0 ** b0, 2.0 ** b1] for (_, b0, b1) in COMBOS])
    s3_dn = s3_dn.view(len(COMBOS), 1, 1, 1, 2, 1)     # (K,1,1,1,2,1)
    s3_up = s3_up.view(len(COMBOS), 1, 1, 1, 2, 1)
    s2_dn = torch.tensor([2.0 ** -a for (a, _, _) in COMBOS]).view(len(COMBOS), 1, 1, 1, 1, 1)
    s2_up = torch.tensor([2.0 ** a for (a, _, _) in COMBOS]).view(len(COMBOS), 1, 1, 1, 1, 1)
    best_err = None
    best_deq = None
    for ci in range(cand.shape[0]):
        e6 = cand[ci]                                      # (N,B)
        rec = (1.0 / e6).view(n, b, 1, 1, 1)
        # (K,N,B,8,2,4)
        vs = g.unsqueeze(0) * rec.unsqueeze(0) * s2_dn * s3_dn
        mant = (torch.round(vs.abs() * 4.0) / 4.0).clamp(0.0, 1.75)
        deq = torch.sign(vs) * mant * s2_up * s3_up * e6.view(1, n, b, 1, 1, 1)
        err = (deq - g.unsqueeze(0)).square().sum((-1, -2))  # (K,N,B,8)
        be, bi = err.min(0)                                  # (N,B,8)
        deq_best = deq.gather(0, bi.view(1, n, b, 8, 1, 1).expand(1, n, b, 8, 2, 4))[0]
        if best_err is None:
            best_err, best_deq = be, deq_best
        else:
            take = be < best_err
            best_err = torch.where(take, be, best_err)
            best_deq = torch.where(take.unsqueeze(-1).unsqueeze(-1), deq_best, best_deq)
    return best_deq.reshape(n, b * 64)


def main():
    ds = ROOT / 'docs/papers/hif4-public-references/test/datasets/public_quantizer/attn.pt'
    g = torch.load(ds, weights_only=True)[0]
    print('roles=%s token_stride=%d' % (ROLES, TOKEN_STRIDE))
    print('%-5s %-4s %11s %11s %9s' % ('toks', 'role', 'heuristic', 'exact_search', 'gain%'))
    agg = {}
    for s in g['test']:
        for role in ROLES:
            x = dequantize_nvfp4(*s[role]).float()
            flat = x.reshape(-1, x.shape[-1])[::TOKEN_STRIDE]
            assert flat.shape[-1] % 64 == 0, flat.shape
            heur = dequantize_hif4(quantize_to_hif4(flat), flat.shape)
            ex = exact_block(flat.reshape(flat.shape[0], -1, 64)).reshape(flat.shape)
            eh = (heur - flat).square().mean().item()
            ee = (ex - flat).square().mean().item()
            print('%-5d %-4s %11.5e %11.5e %8.1f%%'
                  % (flat.shape[0], role, eh, ee, 100 * (1 - ee / eh)), flush=True)
            agg.setdefault(role, []).append((eh, ee))
    print()
    for role, rows in agg.items():
        eh = sum(r[0] for r in rows) / len(rows)
        ee = sum(r[1] for r in rows) / len(rows)
        print('%s 平均: heuristic=%.5e  exact=%.5e  降低 %.1f%%' % (role, eh, ee, 100 * (1 - ee / eh)))


if __name__ == '__main__':
    main()
