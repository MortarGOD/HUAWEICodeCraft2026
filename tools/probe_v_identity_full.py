"""V identity-metric ceiling at FULL bank scale.

Two things make this the one remaining legal lever worth measuring carefully:

  * V needs no counterpart, and its calibration data is real, so a V-side gain is not blocked
    by the fake-counterpart protocol the way fitted Q/K is.
  * the v628 receipt (-8987) showed the judge's V metric is close to the identity, so the
    identity objective we already use is the right one - no P-weighting needed.

probe_v_pool measured the identity oracle on 6 groups only and got +147, but this session has
shown five times that a subset is not evidence (public_quantizer, mini_sample, the 16858
coincidence, v631, and the lambda sweep where a 4-group probe was off by an order of magnitude).
So this runs the whole bank.

Oracle: move each V code +-STEPS on the S1P2 lattice to minimise ||V_hat - V||^2, clamped to the
legal code range. V is never rotated, so the deployed tensor and the exact tensor share a frame
(unlike Q/K, where a frame mismatch invalidated an earlier probe of mine).
"""
import argparse
import contextlib
import io
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import load_solution, gqa_attention, mse, score  # noqa: E402
from hif4_quant import dequantize_nvfp4, dequantize_hif4, quantize_to_hif4  # noqa: E402



def peakedness(raw, qh, kh, dim):
    """Row-normalised attention entropy (0 = one-hot, 1 = uniform) and mean max probability.

    This is measured on the REFERENCE attention (exact q,k), not the player's, so it is a
    property of the bank rather than of our encoding.
    """
    import math
    q = raw['q'].view(-1, qh, dim).transpose(0, 1)
    k = raw['k'].view(-1, kh, dim).transpose(0, 1).repeat_interleave(qh // kh, 0)
    p = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (dim ** 0.5), dim=-1)
    T = p.shape[-1]
    ent = float(-(p * p.clamp_min(1e-12).log()).sum(-1).mean())
    ent = ent / math.log(T) if T > 1 else 0.0
    return ent, float(p.max(-1).values.mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solution', default=str(ROOT / 'solution/solution_v637.py'))
    ap.add_argument('--data', required=True)
    ap.add_argument('--groups', type=int, default=50)
    ap.add_argument('--steps', type=int, default=3)
    ap.add_argument('--question-count', type=int, default=250)
    a = ap.parse_args()
    torch.set_num_threads(int(__import__('os').environ.get('EVAL_THREADS', '4')))
    sol = load_solution(Path(a.solution).resolve())
    groups = torch.load(a.data, map_location='cpu', weights_only=False)[:a.groups]
    dep_s, orc_s = [], []
    ent_s, pmax_s = [], []
    for g in groups:
        qh, kh, dim = g['q_num_heads'], g['kv_num_heads'], g['head_dim']
        with contextlib.redirect_stdout(io.StringIO()):
            states = sol.hif4_calibration_attention(g['calib'], qh, kh, dim)
        for s in g['test']:
            raw = {r: dequantize_nvfp4(*s[r]).float() for r in 'qkv'}
            ref = gqa_attention(raw['q'], raw['k'], raw['v'], qh, kh, dim)
            base = {r: dequantize_hif4(quantize_to_hif4(raw[r]), raw[r].shape).float()
                    for r in 'qkv'}
            bm = mse(ref, gqa_attention(base['q'], base['k'], base['v'], qh, kh, dim))
            play = {}
            for role in 'qkv':
                fn = getattr(sol, 'hif4_dynamic_quantize_' + role)
                num = qh if role == 'q' else kh
                with contextlib.redirect_stdout(io.StringIO()):
                    p = fn(*s[role], num, dim, states[role + '_state'])
                play[role] = dequantize_hif4(p, raw[role].shape).float()
            vp = sol.hif4_dynamic_quantize_v(*s['v'], kh, dim, states['v_state'])
            unit = vp['scale_lv2'] * vp['scale_lv3'] * vp['scale_factor']
            unit = unit.expand(*unit.shape[:-1], 4).flatten(-4, -1).float()
            step = (unit * 0.25).expand_as(play['v'])
            shift = torch.clamp(raw['v'] - play['v'], -a.steps * step, a.steps * step)
            lim = 1.75 * unit
            orc = torch.clamp(play['v'] + torch.round(shift / step) * step, -lim, lim)

            def out(vv):
                return gqa_attention(play['q'], play['k'], vv, qh, kh, dim)

            e, pm = peakedness(raw, qh, kh, dim)
            ent_s.append(e)
            pmax_s.append(pm)
            dep_s.append(score(bm, mse(ref, out(play['v']))))
            orc_s.append(score(bm, mse(ref, out(orc))))
    n = len(dep_s)
    d = sum(dep_s) / n
    o = sum(orc_s) / n
    t = sum(dep_s) / n
    print('%s  groups=%d cases=%d steps=%d' % (Path(a.data).parent.name, len(groups), n, a.steps))
    print('  部署 V            mean %.6f  (250 题 %8.1f)' % (d, d * a.question_count * 100))
    print('  恒等 oracle V     mean %.6f  (250 题 %8.1f)   Δ %+8.1f'
          % (o, o * a.question_count * 100, (o - d) * a.question_count * 100))
    print('  参考注意力 熵/归一化 = %.4f   mean max p = %.4f' % (sum(ent_s) / n, sum(pmax_s) / n))


if __name__ == '__main__':
    main()
