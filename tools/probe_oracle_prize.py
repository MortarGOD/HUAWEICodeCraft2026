"""Size the prize for an instance-level-counterpart Q/K encoder, in GRADED OUTPUT points.

probe_logits_vs_mse shows the logits-MSE headroom but that is not what is graded. This converts it:
per case it takes the deployed codes, runs the same monotone +-1-code-step search against the true
logits, substitutes the resulting Q and K back into the same attention, and reports the output-score
gain. Averaged over every case of the bank, so a single unrepresentative case cannot drive it.

Caveat carried with the number: the oracle uses the TEST INSTANCE's counterpart codes and the true
logits. That is exactly the information an instance-level-counterpart encoder would need, so this is
the ceiling of that family and not of any encoder we can ship. When T is small relative to the code
count the fit is overparameterised and the logits can be matched almost exactly, which inflates the
ceiling at small T; the per-case T is printed so that this can be seen.
"""
import argparse
import contextlib
import importlib.util
import io
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import load_solution, gqa_attention, mse, score  # noqa: E402
from hif4_quant import dequantize_nvfp4, dequantize_hif4, quantize_to_hif4  # noqa: E402

spec = importlib.util.spec_from_file_location('plm', ROOT / 'tools/probe_logits_vs_mse.py')
plm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solution', default=str(ROOT / 'solution/solution_v435.py'))
    ap.add_argument('--data', required=True)
    ap.add_argument('--groups', type=int, default=0)
    ap.add_argument('--sweeps', type=int, default=3)
    ap.add_argument('--question-count', type=int, default=250)
    a = ap.parse_args()
    torch.set_num_threads(int(__import__('os').environ.get('EVAL_THREADS', '4')))
    sol = load_solution(Path(a.solution).resolve())
    groups = torch.load(a.data, map_location='cpu', weights_only=False)
    if a.groups:
        groups = groups[:a.groups]
    rows = []
    for g in groups:
        qh, kh, dim = g['q_num_heads'], g['kv_num_heads'], g['head_dim']
        rep = qh // kh
        with contextlib.redirect_stdout(io.StringIO()):
            states = sol.hif4_calibration_attention(g['calib'], qh, kh, dim)
        for sample in g['test']:
            raw = {r: dequantize_nvfp4(*sample[r]).float() for r in 'qkv'}
            ref = gqa_attention(raw['q'], raw['k'], raw['v'], qh, kh, dim)
            base = {r: dequantize_hif4(quantize_to_hif4(raw[r]), raw[r].shape).float()
                    for r in 'qkv'}
            bm = mse(ref, gqa_attention(base['q'], base['k'], base['v'], qh, kh, dim))
            dep, units = plm.deployed(sol, sample, states, qh, kh, dim)
            T = raw['q'].shape[0]
            q0 = raw['q'].reshape(T, qh, dim).transpose(0, 1).contiguous()
            k0 = (raw['k'].reshape(T, kh, dim).transpose(0, 1).contiguous()
                  .repeat_interleave(rep, 0))
            target = torch.einsum('htd,hud->htu', q0, k0)
            qd = dep['q'].reshape(T, qh, dim).transpose(0, 1).contiguous()
            kd = (dep['k'].reshape(T, kh, dim).transpose(0, 1).contiguous()
                  .repeat_interleave(rep, 0))
            sq = plm.to_htd(units['q'], T, qh, dim)
            sk = plm.to_htd(units['k'], T, kh, dim)
            q1 = plm.greedy(qd.clone(), kd, sq, target, a.sweeps)
            ks1 = plm.greedy_k(kd[::rep].clone(), q1, sk, target, a.sweeps, rep)

            def flat(x):
                return x.transpose(0, 1).reshape(T, -1).contiguous()

            k1 = ks1.repeat_interleave(rep, 0)
            s_dep = score(bm, mse(ref, gqa_attention(dep['q'], dep['k'], dep['v'], qh, kh, dim)))
            s_q = score(bm, mse(ref, gqa_attention(flat(q1), dep['k'], dep['v'], qh, kh, dim)))
            s_qk = score(bm, mse(ref, gqa_attention(flat(q1), flat(ks1), dep['v'], qh, kh, dim)))
            rows.append((T, s_dep, s_q, s_qk))
    n = len(rows)
    print('%s  groups=%d cases=%d' % (Path(a.data).parent.name, len(groups), n))
    print('  T: min=%d med=%d max=%d' % (min(r[0] for r in rows),
                                         int(statistics.median([r[0] for r in rows])),
                                         max(r[0] for r in rows)))
    for i, name in ((1, '部署 Q/K/V'), (2, 'oracle Q'), (3, 'oracle Q+K')):
        v = sum(r[i] for r in rows) / n
        print('  %-12s mean score %+.6f  (%8.1f / %d 题)' % (name, v, v * a.question_count * 100,
                                                             a.question_count))
    d_dep = sum(r[1] for r in rows) / n
    for i, name in ((2, 'oracle Q'), (3, 'oracle Q+K')):
        v = sum(r[i] for r in rows) / n
        print('  Δ %-10s %+8.1f' % (name, (v - d_dep) * a.question_count * 100))


if __name__ == '__main__':
    main()
