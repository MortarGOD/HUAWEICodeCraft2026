"""Attention error budget: is the residual on the Q/K side or the V side?

openaddr's strategy doc reports, from their own judge measurements, that the attention
residual is 71-85% Q/K-side and only 16-30% V-side. Our own measurement on public_quantizer
said the opposite (V ~78%). This decides which role to work on, so it is measured here on the
bank that is supposed to mirror the online layout.

Method (valid as MSE, no coordinate guessing): hold the solution's calibration fixed and swap
one role's quantised tensor for the exact one, scoring against the same standard-HiF4 baseline
the grader uses.

  total      MSE(out(qh,kh,vh), out(q,k,v))
  V side     MSE(out(qh,kh,vh), out(qh,kh,v))    same qh,kh, only v differs
  Q/K side   MSE(out(qh,kh,v ), out(q,k,v))      same v exact, only q,k differ

A bank is used only to size the pool; it is never evidence for an online claim.
"""
import argparse
import io
import contextlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import load_solution, gqa_attention, mse, score  # noqa: E402
from hif4_quant import dequantize_nvfp4, dequantize_hif4, quantize_to_hif4  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solution', default=str(ROOT / 'solution/solution_v435.py'))
    ap.add_argument('--data',
                    default=str(ROOT / 'local_fitted_models_attention_feedback_v2/attn.pt'))
    ap.add_argument('--groups', type=int, default=5)
    ap.add_argument('--question-count', type=int, default=250)
    a = ap.parse_args()
    torch.set_num_threads(4)
    solution = load_solution(Path(a.solution).resolve())
    groups = torch.load(a.data, map_location='cpu', weights_only=False)[:a.groups]

    agg = {k: 0.0 for k in ('base', 'total', 'vside', 'qkside')}
    ncase = 0
    for group in groups:
        qh, kh, dim = group['q_num_heads'], group['kv_num_heads'], group['head_dim']
        with contextlib.redirect_stdout(io.StringIO()):
            states = solution.hif4_calibration_attention(group['calib'], qh, kh, dim)
        for sample in group['test']:
            raw = {r: dequantize_nvfp4(*sample[r]).float() for r in 'qkv'}
            ref = gqa_attention(raw['q'], raw['k'], raw['v'], qh, kh, dim)
            base = {r: dequantize_hif4(quantize_to_hif4(raw[r]), raw[r].shape).float()
                    for r in 'qkv'}
            bm = mse(ref, gqa_attention(base['q'], base['k'], base['v'], qh, kh, dim))
            play = {}
            for role in 'qkv':
                fn = getattr(solution, 'hif4_dynamic_quantize_' + role)
                num = qh if role == 'q' else kh
                with contextlib.redirect_stdout(io.StringIO()):
                    params = fn(*sample[role], num, dim, states[role + '_state'])
                play[role] = dequantize_hif4(params, raw[role].shape).float()
            out_all = gqa_attention(play['q'], play['k'], play['v'], qh, kh, dim)
            out_vx = gqa_attention(play['q'], play['k'], raw['v'], qh, kh, dim)
            agg['base'] += bm
            agg['total'] += mse(ref, out_all)
            agg['vside'] += mse(out_all, out_vx)
            agg['qkside'] += mse(out_vx, ref)
            ncase += 1
    for k in agg:
        agg[k] /= ncase
    s_all = score(agg['base'], agg['total'])
    print('%s  groups=%d cases=%d' % (Path(a.data).name, len(groups), ncase))
    print('  baseline MSE            = %.4e' % agg['base'])
    print('  player  MSE             = %.4e   score=%.6f  (→ %d 题合计 %.1f 分)'
          % (agg['total'], s_all, a.question_count, s_all * a.question_count * 100))
    print('  V 端归因 (MSE)          = %.4e   score=%.6f  → %d 题 %.1f 分'
          % (agg['vside'], score(agg['base'], agg['vside']), a.question_count,
             score(agg['base'], agg['vside']) * a.question_count * 100))
    print('  Q/K 端归因 (MSE)        = %.4e   score=%.6f  → %d 题 %.1f 分'
          % (agg['qkside'], score(agg['base'], agg['qkside']), a.question_count,
             score(agg['base'], agg['qkside']) * a.question_count * 100))
    tot = agg['vside'] + agg['qkside']
    if tot > 0:
        print('  占比： V 端 %.1f%%   Q/K 端 %.1f%%' % (100 * agg['vside'] / tot,
                                                     100 * agg['qkside'] / tot))


if __name__ == '__main__':
    main()
