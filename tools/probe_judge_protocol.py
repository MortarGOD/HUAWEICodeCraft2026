"""Evaluate a candidate under the judge's CONFIRMED calibration protocol.

Every local number in this project so far passed the whole calibration dict (real q, k and v) to a
single hif4_calibration_attention call. The user has confirmed the judge instead calls it once per
role with only the target role real, keeping only that role's state - so on the judge every FITTED
quantity (the rotation decision, the QKS smoothing, the GPTQ Hessian, the refinement) is fitted
through a fake counterpart, while the dynamic path still consumes those states.

That changes which local measurements are meaningful. This runs both protocols and reports the
score under each:

  real   one call, real q/k/v                     (what we have always measured)
  judge  three calls, only the target role real    (the confirmed protocol)

The fake counterpart is built by permuting the token axis of that role inside each sample, with a
fixed seed: shapes are preserved exactly, and the marginal distribution is preserved while the
row-by-row correspondence between roles is destroyed, which is what defeats a fitted correlation.
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


def permute(packed, seed):
    """Permute the token axis of one (quant, scale) pair; shapes are preserved."""
    q, s = packed
    T = q.shape[0]
    if T < 2:
        return packed
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(T, generator=g)
    return [q[idx].contiguous(), s[idx].contiguous()]


def judge_protocol(sol, calib, qh, kh, dim, seed=1234):
    """Three calls, each with only the target role real; keep the matching state."""
    states = {}
    for role in ('q', 'k', 'v'):
        faked = []
        for i, smp in enumerate(calib):
            s2 = dict(smp)
            for other in 'qkv':
                if other != role:
                    s2[other] = permute(smp[other], seed + 97 * i + ord(other))
            faked.append(s2)
        with contextlib.redirect_stdout(io.StringIO()):
            st = sol.hif4_calibration_attention(faked, qh, kh, dim)
        states[role + '_state'] = st[role + '_state']
    return states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solutions', nargs='+',
                    default=[str(ROOT / 'solution/solution_v637.py'),
                             str(ROOT / 'solution/solution_v642.py')])
    ap.add_argument('--data', required=True)
    ap.add_argument('--groups', type=int, default=50)
    ap.add_argument('--question-count', type=int, default=250)
    a = ap.parse_args()
    torch.set_num_threads(int(__import__('os').environ.get('EVAL_THREADS', '4')))
    groups = torch.load(a.data, map_location='cpu', weights_only=False)[:a.groups]
    print('%s  groups=%d' % (Path(a.data).parent.name, len(groups)))
    for path in a.solutions:
        sol = load_solution(Path(path).resolve())
        acc = {'real': [], 'judge': []}
        for g in groups:
            qh, kh, dim = g['q_num_heads'], g['kv_num_heads'], g['head_dim']
            with contextlib.redirect_stdout(io.StringIO()):
                st_real = sol.hif4_calibration_attention(g['calib'], qh, kh, dim)
            st_judge = judge_protocol(sol, g['calib'], qh, kh, dim)
            for s in g['test']:
                raw = {r: dequantize_nvfp4(*s[r]).float() for r in 'qkv'}
                ref = gqa_attention(raw['q'], raw['k'], raw['v'], qh, kh, dim)
                base = {r: dequantize_hif4(quantize_to_hif4(raw[r]), raw[r].shape).float()
                        for r in 'qkv'}
                bm = mse(ref, gqa_attention(base['q'], base['k'], base['v'], qh, kh, dim))
                for tag, states in (('real', st_real), ('judge', st_judge)):
                    play = {}
                    for r in 'qkv':
                        fn = getattr(sol, 'hif4_dynamic_quantize_' + r)
                        num = qh if r == 'q' else kh
                        with contextlib.redirect_stdout(io.StringIO()):
                            p = fn(*s[r], num, dim, states[r + '_state'])
                        play[r] = dequantize_hif4(p, raw[r].shape).float()
                    acc[tag].append(score(bm, mse(
                        ref, gqa_attention(play['q'], play['k'], play['v'], qh, kh, dim))))
        n = len(acc['real'])
        for tag in ('real', 'judge'):
            v = sum(acc[tag]) / n
            print('  %-22s %-6s mean %.6f  (250 题 %8.1f)'
                  % (Path(path).name, tag, v, v * a.question_count * 100), flush=True)


if __name__ == '__main__':
    main()
