"""Where does the V-side pool on the online-profiled bank come from?

The v625 receipt (delta exactly 0) killed the round 8-11 guard line and, with it, the
mini_sample basis for the role structure. On the third-party bank built to match the online
distribution (qwen2_5_14b_online_profiled_v2) V carries ~66% of the attention residual, and V
is the side that needs no counterpart. So the question is whether that pool is reachable by a
legal V encoder or is an artefact of the objective.

Per case this measures, against the same standard-HiF4 baseline the grader uses:

  player          our dynamic V encoder, as deployed
  V = std         V replaced by the standard HiF4 quantiser (isolates what our V buys)
  V = exact       V replaced by the exact tensor (the whole V pool)
  V = oracle-id   our V moved +-STEPS code steps per element to minimise ||V_hat - V||^2
                  (identity metric: a legal objective, needs no counterpart)
  V = oracle-P    our V moved by greedy coordinate descent on the GRADED metric
                  ||P (V_hat - V)||^2 with P the player's own softmax (an oracle: P needs Q/K)

If oracle-id is far below player, a better identity-metric encoder is a legal mechanism.
If oracle-P is far below oracle-id, the pool is a metric mismatch and the legal question
becomes whether a counterpart-free surrogate for P is good enough.

A bank is used only to size a pool; it is never evidence for an online claim.
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


def v_step(p, like):
    unit = p['scale_lv2'] * p['scale_lv3'] * p['scale_factor']
    unit = unit.expand(*unit.shape[:-1], 4).flatten(-4, -1).float()
    return (unit * 0.25).expand_as(like)


def oracle_identity(v, target, step, steps, legal=None):
    """Per-element optimum of the S1P2 lattice within +-steps; exact for the identity metric."""
    shift = torch.clamp(target - v, -steps * step, steps * step)
    out = v + torch.round(shift / step) * step
    if legal is not None:
        out = torch.clamp(out, -legal, legal)
    return out


def oracle_metric(v, target, step, steps, PtP, rep, sweeps=2, legal=None):
    """Exact coordinate descent on the graded metric ||P (V_hat - V)||^2.

    v, target, step: (kh, T, dim). PtP: (qh, T, T) = P^T P per query head; P is the player's
    own softmax. The V error of KV head j is seen by all `rep` query heads of its group, so
    that head's metric is the group sum M_j.

    trace(E^T M E) = sum_c E[:, c]^T M E[:, c], so the columns are independent quadratics and
    a whole token row can be moved for every channel at once exactly. The exact change from
    moving element (t, c) by d is 2 d g + d^2 M[t, t] with g = (M E)[t, c], and moving it
    updates G = M E by M[:, t] * d.
    """
    kh = v.shape[0]
    cur = v.clone()
    metas = [PtP[j * rep:(j + 1) * rep].sum(0) for j in range(kh)]
    G = torch.stack([metas[j] @ (cur[j] - target[j]) for j in range(kh)])
    for _ in range(sweeps):
        moved = 0
        for j in range(kh):
            M = metas[j]
            for t in range(cur.shape[1]):
                h = float(M[t, t])
                if h < 1e-30:
                    continue
                g = G[j, t]
                d = torch.clamp(-g / h, -steps * step[j, t], steps * step[j, t])
                d = torch.round(d / step[j, t]) * step[j, t]
                acc = (2.0 * d * g + d * d * h) < 0
                if not bool(acc.any()):
                    continue
                d = torch.where(acc, d, torch.zeros_like(d))
                if legal is not None:
                    lim = legal[j, t]
                    d = torch.clamp(cur[j, t] + d, -lim, lim) - cur[j, t]
                cur[j, t] = cur[j, t] + d
                G[j] = G[j] + M[:, t][:, None] * d[None, :]
                moved += int(acc.sum())
        if moved == 0:
            break
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solution', default=str(ROOT / 'solution/solution_v435.py'))
    ap.add_argument('--data',
                    default=str(ROOT / 'docs/papers/hif4-public-references/test/datasets/'
                                     'qwen2_5_14b_online_profiled_v2/attn.pt'))
    ap.add_argument('--groups', type=int, default=10)
    ap.add_argument('--steps', type=int, default=3)
    ap.add_argument('--legal', action='store_true',
                    help='clamp moves to the legal mantissa range |v| <= 1.75*unit')
    ap.add_argument('--lambdas', type=float, nargs='*',
                    default=[0.0, 0.25, 1.0, 4.0, 16.0],
                    help='lambda in the counterpart-free metric I + lam*11^T')
    ap.add_argument('--question-count', type=int, default=250)
    a = ap.parse_args()
    torch.set_num_threads(int(__import__('os').environ.get('EVAL_THREADS', '4')))
    sol = load_solution(Path(a.solution).resolve())
    groups = torch.load(a.data, map_location='cpu', weights_only=False)[:a.groups]

    keys = ('player', 'v_std', 'v_exact', 'oracle_id', 'oracle_P') \
        + tuple('sur_%g' % l for l in a.lambdas)
    acc = {k: [] for k in keys}
    for gi, g in enumerate(groups):
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
            step = v_step(vp, play['v'])

            def out_with(vv):
                return gqa_attention(play['q'], play['k'], vv, qh, kh, dim)

            acc['player'].append(score(bm, mse(ref, out_with(play['v']))))
            acc['v_std'].append(score(bm, mse(ref, out_with(base['v']))))
            acc['v_exact'].append(score(bm, mse(ref, out_with(raw['v']))))
            # The legal limit on |value| is 1.75 * unit = 7 * step, since step = 0.25 * unit.
            lim = 7.0 * step
            legal = (None if not a.legal
                     else lim.view(-1, kh, dim).transpose(0, 1).contiguous())
            oid = oracle_identity(play['v'], raw['v'], step, a.steps, lim)
            acc['oracle_id'].append(score(bm, mse(ref, out_with(oid))))
            # graded metric from the player's own probabilities
            qf = play['q'].view(-1, qh, dim).transpose(0, 1)
            kf = play['k'].view(-1, kh, dim).transpose(0, 1).repeat_interleave(qh // kh, 0)
            probs = torch.softmax(torch.matmul(qf, kf.transpose(-1, -2)) / (dim ** 0.5), dim=-1)
            PtP = torch.matmul(probs.transpose(-1, -2), probs)
            v_ktd = play['v'].view(-1, kh, dim).transpose(0, 1).contiguous()
            t_ktd = raw['v'].view(-1, kh, dim).transpose(0, 1).contiguous()
            s_ktd = step.view(-1, kh, dim).transpose(0, 1).contiguous()
            vf_o = oracle_metric(v_ktd, t_ktd, s_ktd, a.steps, PtP, qh // kh, legal=legal)
            acc['oracle_P'].append(score(bm, mse(
                ref, out_with(vf_o.transpose(0, 1).reshape(-1, kh * dim)))))
            # Counterpart-free surrogates: P is a softmax, so P^T P interpolates between the
            # identity (peaked rows) and the rank-one 11^T (uniform rows). Both endpoints are
            # computable from V alone, so a metric of the form I + lam*11^T needs no Q/K.
            T = v_ktd.shape[1]
            eye = torch.eye(T)
            ones = torch.ones(T, T)
            for lam in a.lambdas:
                M = (eye + lam * ones).unsqueeze(0).repeat(qh, 1, 1)
                o = oracle_metric(v_ktd, t_ktd, s_ktd, a.steps, M, qh // kh, legal=legal)
                acc['sur_%g' % lam].append(score(bm, mse(
                    ref, out_with(o.transpose(0, 1).reshape(-1, kh * dim)))))

    n = len(acc['player'])
    print('%s  groups=%d cases=%d  steps=%d'
          % (Path(a.data).parent.name, len(groups), n, a.steps))
    for k in keys:
        v = sum(acc[k]) / n
        print('  %-10s mean score = %.6f   (250 题 %8.1f)' % (k, v, v * a.question_count * 100))
    print('  ---- 差距 ----')
    p = sum(acc['player']) / n
    for k in keys[1:]:
        v = sum(acc[k]) / n
        print('  %-10s - player = %+.6f   (250 题 %+8.1f)' % (k, v - p, (v - p) * a.question_count * 100))


if __name__ == '__main__':
    main()
