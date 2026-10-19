"""How sensitive is the graded attention score to the legal Q/K geometry knob?

The Q/K side carries ~87% of the official sample's attention residual, so if any legal
Q/K lever still has room, the score must at least *react* to it. The only logits-preserving
legal lever is a fixed orthogonal transform applied to both roles; the family currently
deployed is H*D (Hadamard x random sign), whose 64 seeds were already searched with seed 0
winning. This measures the sensitivity to that whole family and to neighbouring choices:

  current      deployed H*D, seed 0
  seed1        same family, different sign pattern
  random-orth  a genuinely different fixed orthogonal (QR of a seeded Gaussian)
  dct          DCT-II, a different structured fixed orthogonal
  q-norot      Q left unrotated, K still rotated   (logits no longer preserved)
  k-norot      K left unrotated, Q still rotated   (logits no longer preserved)
  none-norot   neither rotated = plain base quantiser

A zero here means the attention score cannot be moved by Q/K geometry on that bank; a large
spread means the knob is live and worth spending a probe submission on.
"""
import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import load_solution, evaluate_attention  # noqa: E402


def _orth(dh, seed):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(dh, dh, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    sgn = torch.sign(torch.diagonal(r))
    sgn[sgn == 0] = 1.0
    return (q * sgn.unsqueeze(0)).to(torch.float32)


def _dct(dh):
    n = torch.arange(dh, dtype=torch.float64)
    k = torch.arange(dh, dtype=torch.float64)[:, None]
    c = torch.cos(math.pi * (n[None, :] + 0.5) * k / dh) * math.sqrt(2.0 / dh)
    c[0] = math.sqrt(1.0 / dh)
    return c.to(torch.float32)


def build(solution_path, variant):
    sol = load_solution(Path(solution_path).resolve())
    if variant == 'current':
        return sol
    orig_calib = sol.hif4_calibration_attention

    if variant in ('q-norot', 'k-norot', 'none-norot', 'seed1', 'force-rot'):
        def calib(samples, qh, kh, d, _o=orig_calib, _v=variant):
            st = _o(samples, qh, kh, d)
            if _v in ('q-norot', 'none-norot'):
                st['q_state'] = dict(st['q_state'], rot=0)
            if _v in ('k-norot', 'none-norot'):
                st['k_state'] = dict(st['k_state'], rot=0)
            if _v == 'seed1':
                st['q_state'] = dict(st['q_state'], rseed=1)
                st['k_state'] = dict(st['k_state'], rseed=1)
            if _v == 'force-rot':
                # The deployed search leaves rot=0 whenever no candidate beats the
                # unrotated calibration loss; this forces the Hadamard on regardless.
                st['q_state'] = dict(st['q_state'], rot=1, rseed=0)
                st['k_state'] = dict(st['k_state'], rot=1, rseed=0)
            return st
        sol.hif4_calibration_attention = calib
        return sol

    if variant in ('random-orth', 'dct', 'force-dct', 'force-random'):
        # force-* : the deployed guard rejects the rotation outright on the official sample
        # (its guard sample is T=1, where Q/K provably cannot matter), so a basis swap is only
        # observable once the rotation is actually on.
        if variant.startswith('force-'):
            def calib(samples, qh, kh, d, _o=orig_calib):
                st = _o(samples, qh, kh, d)
                st['q_state'] = dict(st['q_state'], rot=1, rseed=0)
                st['k_state'] = dict(st['k_state'], rot=1, rseed=0)
                return st
            sol.hif4_calibration_attention = calib

        def make_R(dh, seed_index=0, _v=variant):
            if dh & (dh - 1):
                return None                      # match the deployed guard exactly
            return _orth(dh, 0x1234 + dh) if _v.endswith('random') else _dct(dh)
        sol._make_R = make_R
        return sol

    raise SystemExit('unknown variant ' + variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solution', default=str(ROOT / 'solution/solution_v435.py'))
    ap.add_argument('--data', default=str(ROOT / 'mini_sample/attn.pt'))
    ap.add_argument('--groups', type=int, default=1)
    ap.add_argument('--question-count', type=int, default=250)
    ap.add_argument('--variants', default='current,seed1,random-orth,dct,q-norot,k-norot,none-norot')
    a = ap.parse_args()
    torch.set_num_threads(int(__import__('os').environ.get('EVAL_THREADS', '4')))
    groups = torch.load(a.data, map_location='cpu', weights_only=False)[:a.groups]
    print('bank=%s groups=%d  (mean_quality over %d cases)'
          % (Path(a.data).name, len(groups), sum(len(g['test']) for g in groups)))
    base = None
    for v in a.variants.split(','):
        sol = build(a.solution, v)
        scores, secs = evaluate_attention(sol, groups)
        mean = sum(scores) / len(scores)
        if base is None:
            base = mean
        print('  %-12s mean=%.6f  Δ=%+.6f  implied_online=%8.1f  wall=%.1fs'
              % (v, mean, mean - base, mean * a.question_count * 100, secs))


if __name__ == '__main__':
    main()
