"""public_quantizer: what does the exact HiF4 encoder buy, per role and end-to-end?

Part 1  element-wise MSE of the encoder on Q, K and V (heuristic vs exact).
Part 2  deployed attention-output MSE with Q and K held at the v441 solution path and only
        the V encoder swapped, plus the resulting quality / local points / predicted online.
Part 3  wall time of the exact encoder, because it must run on the dynamic path.
"""
import copy
import sys
import time
from pathlib import Path

import torch

ROOT = Path('/Users/ee/Code/2026-huawei-alogorithm')
sys.path.insert(0, str(ROOT))
from hif4_quant import dequantize_nvfp4, dequantize_hif4, quantize_to_hif4  # noqa: E402
from tools.test_v422 import load  # noqa: E402
from tools.exact_hif4_core import exact_hif4_dequant, exact_hif4_params  # noqa: E402
from evaluate_local import gqa_attention, mse, score  # noqa: E402

torch.set_num_threads(4)
torch.manual_seed(20260910)
VERSION = 441
DS = ROOT / 'docs/papers/hif4-public-references/test/datasets/public_quantizer/attn.pt'


@torch.inference_mode()
def main():
    g = torch.load(DS, weights_only=True)[0]
    qh, kh, d = g['q_num_heads'], g['kv_num_heads'], g['head_dim']

    print('=== Part 1: element-wise encoder MSE (heuristic vs exact) ===')
    print('%-5s %-4s %12s %12s %9s' % ('toks', 'role', 'heuristic', 'exact', 'gain%'))
    agg = {}
    for s in g['test']:
        for role in 'qkv':
            x = dequantize_nvfp4(*s[role]).float().reshape(-1, dequantize_nvfp4(*s[role]).shape[-1])
            heur = dequantize_hif4(quantize_to_hif4(x), x.shape)
            ex = exact_hif4_dequant(x)
            eh = (heur - x).square().mean().item()
            ee = (ex - x).square().mean().item()
            print('%-5d %-4s %12.5e %12.5e %8.1f%%' % (x.shape[0], role, eh, ee, 100 * (1 - ee / eh)))
            agg.setdefault(role, []).append((eh, ee))
    for role, rows in agg.items():
        eh = sum(r[0] for r in rows) / len(rows)
        ee = sum(r[1] for r in rows) / len(rows)
        print('%s 平均: heuristic=%.5e exact=%.5e  降低 %.1f%%' % (role, eh, ee, 100 * (1 - ee / eh)))

    print()
    print('=== Part 2: deployed attention MSE, Q/K fixed at v%d ===' % VERSION)
    cal = load(VERSION, 'calib_q_0').hif4_calibration_attention(copy.deepcopy(g['calib']), qh, kh, d)
    mods = {r: load(VERSION, 'dyn_%s_0' % r) for r in 'qkv'}
    print('%-5s %11s %11s %11s %11s %11s %9s' % ('toks', 'baseHiF4', 'v441', 'exactV',
                                                 'exactQK', 'exactQKV', 'q_gain%'))
    rows = []
    for s in g['test']:
        raw = {r: dequantize_nvfp4(*s[r]).float() for r in 'qkv'}
        tgt = gqa_attention(raw['q'], raw['k'], raw['v'], qh, kh, d)
        b = {r: dequantize_hif4(quantize_to_hif4(raw[r]), raw[r].shape) for r in 'qkv'}
        base_mse = mse(tgt, gqa_attention(b['q'], b['k'], b['v'], qh, kh, d))
        dq = {}
        for role in 'qkv':
            st = copy.deepcopy(cal[role + '_state'])
            fn = getattr(mods[role], 'hif4_dynamic_quantize_' + role)
            dq[role] = dequantize_hif4(fn(*s[role], qh if role == 'q' else kh, d, st), raw[role].shape)
        m441 = mse(tgt, gqa_attention(dq['q'], dq['k'], dq['v'], qh, kh, d))
        ex = {r: exact_hif4_dequant(raw[r].reshape(-1, raw[r].shape[-1])).reshape(raw[r].shape)
              for r in 'qkv'}
        mv = mse(tgt, gqa_attention(dq['q'], dq['k'], ex['v'], qh, kh, d))
        mqk = mse(tgt, gqa_attention(ex['q'], ex['k'], dq['v'], qh, kh, d))
        mall = mse(tgt, gqa_attention(ex['q'], ex['k'], ex['v'], qh, kh, d))
        rows.append((base_mse, m441, mv, mqk, mall))
        print('%-5d %11.4e %11.4e %11.4e %11.4e %11.4e %8.1f%%'
              % (raw['q'].shape[0], base_mse, m441, mv, mqk, mall,
                 100 * (m441 - mall) / m441), flush=True)
    n = len(rows)
    bm = sum(r[0] for r in rows) / n
    for label, idx, ref in (('v441', 1, None), ('exact-V', 2, 1), ('exact-QK', 3, 1), ('exact-QKV', 4, 1)):
        mm = sum(r[idx] for r in rows) / n
        q = score(bm, mm)
        line = '%-10s player MSE = %.4e  quality = %.6f  local = %.1f' % (label, mm, q, q * 25000)
        if ref is not None:
            mr = sum(r[ref] for r in rows) / n
            dl = (q - score(bm, mr)) * 25000
            line += '   Δlocal = %+.1f  -> 预计线上 %+.0f' % (dl, 1.294 * dl)
        print(line)

    print()
    print('=== Part 3: exact encoder wall time (dynamic path cost) ===')
    for toks in (10, 128, 512, 1024):
        x = torch.randn(toks, kh * d)
        t0 = time.perf_counter()
        exact_hif4_params(x)
        t1 = time.perf_counter()
        q = torch.randn(toks, qh * d)
        t2 = time.perf_counter()
        exact_hif4_params(q)
        t3 = time.perf_counter()
        print('T=%-5d V(512) %.3fs   Q(4096) %.3fs' % (toks, t1 - t0, t3 - t2))


if __name__ == '__main__':
    main()
