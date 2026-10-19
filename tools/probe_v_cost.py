"""How concentrated is v628's gain across token rows, and what would a cheaper screen cost?

v628 visits every token row that has any beneficial move, and the measured cost is +14..+27s
on the mirror's 250 cases (est. 278..291s online against a 300s limit). The sequential body is
the whole cost, so the question is whether a small subset of rows carries most of the gain.

This runs the parent v435's V encoder and then the rank-1 pass by hand, visiting only the rows
whose per-row benefit is above a quantile. Report: retained gain and the fraction of rows
visited. A bank sizes the trade-off only.
"""
import argparse
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import load_solution, gqa_attention, mse, score  # noqa: E402
from hif4_quant import dequantize_nvfp4, dequantize_hif4, quantize_to_hif4  # noqa: E402

spec = importlib.util.spec_from_file_location('pool', ROOT / 'tools/probe_v_pool.py')
pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool)

LAM = 0.1
STEPS = 3


def rank1(params, target, visit_mask=None, sweeps=1):
    """v628's body, optionally visiting only the rows selected by visit_mask."""
    sign = params['sign']
    unit = params['scale_lv2'] * params['scale_lv3'] * params['scale_factor']
    step = unit * 0.25
    val = sign * params['mant'] * unit
    target = target.reshape(val.shape)
    err = val - target
    colsum = err.sum(0, keepdim=True)
    h = 1.0 + LAM
    g0 = err + LAM * colsum
    d0 = torch.clamp(-g0 / h, -STEPS * step, STEPS * step)
    d0 = torch.round(d0 / step) * step
    benefit = -(2.0 * d0 * g0 + d0 * d0 * h)          # positive where a move helps
    per_row = benefit.clamp_min(0.0).sum(dim=tuple(range(1, benefit.dim())))
    order = torch.argsort(per_row, descending=True)
    if visit_mask is not None:
        rows = order[:max(1, int(round(visit_mask * order.numel())))]
    else:
        rows = order
    for t in rows.tolist():
        g = err[t] + LAM * colsum[0]
        delta = torch.clamp(-g / h, -STEPS * step[t], STEPS * step[t])
        delta = torch.round(delta / step[t]) * step[t]
        delta = torch.where((2.0 * delta * g + delta * delta * h) < 0.0,
                            delta, torch.zeros_like(delta))
        lim = 1.75 * unit[t]
        lo = torch.minimum(torch.zeros_like(lim), lim * sign[t])
        hi = torch.maximum(torch.zeros_like(lim), lim * sign[t])
        moved = torch.clamp(val[t] + delta, lo, hi)
        delta = moved - val[t]
        val[t] = moved
        err[t] = moved - target[t]
        colsum[0] = colsum[0] + delta
    out = dict(params)
    out['mant'] = torch.round(val.abs() / unit * 4.0).clamp(0.0, 7.0) / 4.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solution', default=str(ROOT / 'solution/solution_v435.py'))
    ap.add_argument('--data',
                    default=str(ROOT / 'local_fitted_models_attention_feedback_v2/attn.pt'))
    ap.add_argument('--groups', type=int, default=6)
    ap.add_argument('--fracs', type=float, nargs='*', default=[1.0, 0.5, 0.25, 0.1])
    a = ap.parse_args()
    torch.set_num_threads(int(__import__('os').environ.get('EVAL_THREADS', '4')))
    sol = load_solution(Path(a.solution).resolve())
    groups = torch.load(a.data, map_location='cpu', weights_only=False)[:a.groups]
    acc = {f: [] for f in a.fracs}
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
            for f in a.fracs:
                q = rank1(vp, raw['v'], visit_mask=f)
                vv = dequantize_hif4(q, raw['v'].shape).float()
                acc[f].append(score(bm, mse(
                    ref, gqa_attention(play['q'], play['k'], vv, qh, kh, dim))))
    n = len(acc[a.fracs[0]])
    print('%s  groups=%d cases=%d' % (Path(a.data).parent.name, len(groups), n))
    full = sum(acc[1.0]) / n
    for f in a.fracs:
        v = sum(acc[f]) / n
        print('  访问行比例 %4.0f%%   平均分 %.6f   Δ vs 全访问 %+8.1f (250 题)'
              % (100 * f, v, (v - full) * 25000))


if __name__ == '__main__':
    main()
