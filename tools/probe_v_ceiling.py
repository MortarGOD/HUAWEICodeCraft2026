"""On a T=1 attention case the output is exactly V, so the score is decided by V alone.

The official token list is [10,128,512,1024,1], so if the judge follows it a fifth of the
attention questions are pure-V questions where Q/K provably cannot matter. That makes the
V side a legal, counterpart-free pool on exactly those questions: V's calibration data is
real and its target is V itself.

This measures, per official test case:
  baseline   standard HiF4 of the exact V, the grader's reference standard
  deployed   our dynamic V encoder
  oracle     per-element best point of the S1P2 lattice within +-STEPS code steps,
             i.e. the ceiling of any encoder that may move each element by whole codes

For the T=1 case the score is exactly 1 - MSE(V_hat, V)/MSE(V_std, V), so the oracle row is
the ceiling of the whole T=1 question. It is an over-estimate by construction (it ignores the
mantissa range limit and any structure the encoder must keep), which is the safe direction for
deciding whether a pool exists at all. A local bank is used only to size the pool.
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import load_solution, mse, score  # noqa: E402
from hif4_quant import dequantize_nvfp4, dequantize_hif4, quantize_to_hif4  # noqa: E402

STEPS = 3


def unit_step(p, role_flat):
    """One S1P2 code step per element: 0.25 x (shared scales), expanded to the flat layout."""
    unit = p['scale_lv2'] * p['scale_lv3'] * p['scale_factor']
    unit = unit.expand(*unit.shape[:-1], 4).flatten(-4, -1).float()
    return (unit * 0.25).expand_as(role_flat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solution', default=str(ROOT / 'solution/solution_v626.py'))
    ap.add_argument('--data', default=str(ROOT / 'mini_sample/attn.pt'))
    ap.add_argument('--steps', type=int, default=STEPS)
    a = ap.parse_args()
    torch.set_num_threads(int(__import__('os').environ.get('EVAL_THREADS', '4')))
    sol = load_solution(Path(a.solution).resolve())
    g = torch.load(a.data, map_location='cpu', weights_only=False)[0]
    qh, kh, dim = g['q_num_heads'], g['kv_num_heads'], g['head_dim']
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        states = sol.hif4_calibration_attention(g['calib'], qh, kh, dim)

    print('%-8s %-14s %-14s %-14s %-9s %-9s %-9s'
          % ('case', 'baseline MSE', 'deployed MSE', 'oracle MSE', 'score', 'oracle-s', 'ratio'))
    acc = []
    for i, s in enumerate(g['test']):
        T = int(s['v'][0].shape[0])
        raw = dequantize_nvfp4(*s['v']).float()
        base = dequantize_hif4(quantize_to_hif4(raw), tuple(raw.shape)).float()
        p = sol.hif4_dynamic_quantize_v(*s['v'], kh, dim, states['v_state'])
        dep = dequantize_hif4(p, tuple(raw.shape)).float()
        step = unit_step(p, dep)
        shift = torch.clamp(raw - dep, -a.steps * step, a.steps * step)
        orc = dep + torch.round(shift / step) * step
        e_b, e_d, e_o = mse(raw, base), mse(raw, dep), mse(raw, orc)
        acc.append((T, e_b, e_d, e_o))
        print('%-8d %-14.5e %-14.5e %-14.5e %-9.6f %-9.6f %-9.4f'
              % (i, e_b, e_d, e_o, score(e_b, e_d), score(e_b, e_o), e_d / e_b))
    print()
    print('T=1 是纯 V 题：score 就是 1 - MSE(V_hat,V)/MSE(V_std,V)')
    for T, e_b, e_d, e_o in acc:
        if T == 1:
            print('  T=1 当前 score=%.6f   oracle 上限=%.6f   → 单题余量 %.6f'
                  % (score(e_b, e_d), score(e_b, e_o), score(e_b, e_o) - score(e_b, e_d)))
    sb = sum(r[1] for r in acc) / len(acc)
    sd = sum(r[2] for r in acc) / len(acc)
    so = sum(r[3] for r in acc) / len(acc)
    print('  5 例均值（聚合 MSE，仅作参考，评分是逐例再平均）:'
          ' 当前 %.6f  oracle %.6f' % (score(sb, sd), score(sb, so)))


if __name__ == '__main__':
    main()
