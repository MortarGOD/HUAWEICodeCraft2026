"""Is there anything to gain by choosing Q/K codes for the LOGITS instead of per-tensor MSE?

Our base quantiser minimises each tensor's own reconstruction error. The graded objective is the
attention output, whose immediate driver is the logits Q K^T, so the question is whether a code
search that optimises the logits directly could beat the deployed codes.

Method: take the deployed (rotated-coordinate) Q_hat/K_hat exactly as the solution returns them,
and run a greedy coordinate descent over the S1P2 lattice (+/- one code step per element) that
minimises ||Q_hat K_hat^T - Q K^T||^2. That is an achievable upper bound for a logits-aware
encoder, not a lower bound, and it is measured against the same target the deployed codes face.

A local bank is used only to size this gap; it is never evidence for an online claim.
"""
import argparse
import copy
import io
import contextlib
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import load_solution, gqa_attention, mse, score  # noqa: E402
from hif4_quant import quantize_to_hif4, dequantize_hif4  # noqa: E402
from hif4_quant import dequantize_nvfp4  # noqa: E402


def deployed(solution, sample, states, qh, kh, dim):
    out = {}
    units = {}
    for role in ('q', 'k', 'v'):
        fn = getattr(solution, 'hif4_dynamic_quantize_' + role)
        with contextlib.redirect_stdout(io.StringIO()):
            p = fn(*sample[role], qh if role == 'q' else kh, dim, states[role + '_state'])
        val = (p['sign'] * p['mant'] * p['scale_lv2'] * p['scale_lv3']
               * p['scale_factor']).flatten(-4, -1).float()
        unit = (p['scale_lv2'] * p['scale_lv3'] * p['scale_factor'])
        unit = unit.expand(*unit.shape[:-1], 4).flatten(-4, -1).float()
        out[role] = val
        units[role] = unit * 0.25
    return out, units


def to_htd(flat, t, heads, dim):
    return flat.reshape(t, heads, dim).transpose(0, 1).contiguous()


def greedy(x, other, step, target, sweeps):
    """Monotone row-block descent on ||x@other^T-target||^2 using +/- one code step.

    A whole (head,row) block of channels is proposed at once (the clipped Newton direction for
    the quadratic), then a halving line search accepts the proposal only if the exact block error
    decreases. Block proposals keep it vectorised; the line search keeps it monotone, which the
    naive per-channel Jacobi sweep was not (it ignored the cross-channel Gram terms and diverged).
    """
    heads, t, dim = x.shape
    resid = torch.einsum('htd,hud->htu', x, other) - target
    for _ in range(sweeps):
        moved = 0
        for h in range(heads):
            oh = other[h]
            w = (oh * oh).sum(0).clamp_min(1e-30)
            for i in range(t):
                r = resid[h, i]
                e = float(r.square().sum())
                g = r @ oh
                d = torch.clamp(-g / w, -step[h, i], step[h, i])
                for _try in range(6):
                    cand = r + oh @ d
                    if float(cand.square().sum()) < e:
                        x[h, i] = x[h, i] + d
                        resid[h, i] = cand
                        moved += 1
                        break
                    d = d * 0.5
        if moved == 0:
            break
    return x


def greedy_k(ks, q, step, target, sweeps, g):
    """K-side descent, tied across each GQA group.

    The K pass cannot reuse ``greedy``. There ``x`` is the first factor, so it would fit
    ``k1 @ q1^T`` against ``target``; the graded quantity is ``q1 @ k1^T``, and the two
    objectives coincide only for a symmetric target. Working in the transposed frame
    (``ks[j] @ q[h]^T`` vs ``target[h]^T``) has the identical Frobenius norm and does fit
    the objective we measure.

    ``ks`` holds one row per *KV* head. GQA repeats each row across the query heads of its
    group, so copies moved independently are not a K tensor any legal encoder can emit.
    Each KV row therefore takes a single step, driven by the gradient summed over its group
    and normalised by the summed per-channel curvature of that group.
    """
    kh, t, dim = ks.shape
    for _ in range(sweeps):
        moved = 0
        for j in range(kh):
            grp = q[j * g:(j + 1) * g]                              # (g, t, dim)
            tgt = target[j * g:(j + 1) * g].transpose(-1, -2)       # (g, t, t)
            r_all = torch.einsum('td,gud->gtu', ks[j], grp) - tgt
            w = (grp * grp).sum(dim=(0, 1)).clamp_min(1e-30)        # diag of the block Hessian
            for i in range(t):
                r = r_all[:, i, :]
                e = float(r.square().sum())
                grad = torch.einsum('gu,gud->d', r, grp)
                d = torch.clamp(-grad / w, -step[j, i], step[j, i])
                for _try in range(6):
                    cand = r + torch.einsum('d,gud->gu', d, grp)
                    if float(cand.square().sum()) < e:
                        ks[j, i] = ks[j, i] + d
                        r_all[:, i, :] = cand
                        moved += 1
                        break
                    d = d * 0.5
        if moved == 0:
            break
    return ks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solution', default=str(ROOT / 'solution/solution_v435.py'))
    ap.add_argument('--data', default=str(ROOT / 'mini_sample/attn.pt'))
    ap.add_argument('--groups', type=int, default=1)
    ap.add_argument('--case', type=int, default=1)
    ap.add_argument('--sweeps', type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('EVAL_THREADS', '4')))
    solution = load_solution(Path(a.solution).resolve())
    groups = torch.load(a.data, map_location='cpu', weights_only=False)[:a.groups]
    g = groups[0]
    qh, kh, dim = g['q_num_heads'], g['kv_num_heads'], g['head_dim']
    with contextlib.redirect_stdout(io.StringIO()):
        states = solution.hif4_calibration_attention(g['calib'], qh, kh, dim)
    sample = g['test'][a.case]
    raw = {r: dequantize_nvfp4(*sample[r]).float() for r in 'qkv'}
    T = raw['q'].shape[0]
    q0 = raw['q'].reshape(T, qh, dim).transpose(0, 1).contiguous()
    k0 = raw['k'].reshape(T, kh, dim).transpose(0, 1).contiguous().repeat_interleave(qh // kh, 0)
    target = torch.einsum('htd,hud->htu', q0, k0)                    # exact logits, rotated frame

    dep, units = deployed(solution, sample, states, qh, kh, dim)
    qd = dep['q'].reshape(T, qh, dim).transpose(0, 1).contiguous()
    kd = dep['k'].reshape(T, kh, dim).transpose(0, 1).contiguous().repeat_interleave(qh // kh, 0)
    e0 = (torch.einsum('htd,hud->htu', qd, kd) - target).square().mean().item()

    sq = to_htd(units['q'], T, qh, dim)
    sk = to_htd(units['k'], T, kh, dim)
    grp = qh // kh
    ks_dep = kd[::grp].contiguous()          # the g copies of a KV head are adjacent

    q1 = greedy(qd.clone(), kd, sq, target, a.sweeps)
    e1 = (torch.einsum('htd,hud->htu', q1, kd) - target).square().mean().item()

    ks_only = greedy_k(ks_dep.clone(), qd, sk, target, a.sweeps, grp)
    ek = (torch.einsum('htd,hud->htu', qd, ks_only.repeat_interleave(grp, 0))
          - target).square().mean().item()

    ks1 = greedy_k(ks_dep.clone(), q1, sk, target, a.sweeps, grp)
    k1 = ks1.repeat_interleave(grp, 0)
    e2 = (torch.einsum('htd,hud->htu', q1, k1) - target).square().mean().item()

    # The tied pass optimises the transposed objective; it must evaluate to the very same
    # number as the graded one, otherwise the K tensor it produced is not the one measured.
    fwd = (torch.einsum('htd,hud->htu', q1, k1) - target).square().sum().item()
    trs = 0.0
    for j in range(kh):
        grp_j = q1[j * grp:(j + 1) * grp]
        trs += float((torch.einsum('td,gud->gtu', ks1[j], grp_j)
                      - target[j * grp:(j + 1) * grp].transpose(-1, -2)).square().sum())
    rel = abs(fwd - trs) / max(fwd, 1e-30)

    # What the logits-MSE reduction is WORTH in the graded output score. The oracle codes live in
    # the deployed (rotated) frame, and the reference/baseline are frame-independent because the
    # rotation preserves the logits, so substituting them straight into the same attention is valid.
    ref = gqa_attention(dequantize_nvfp4(*sample['q']).float(),
                        dequantize_nvfp4(*sample['k']).float(),
                        dequantize_nvfp4(*sample['v']).float(), qh, kh, dim)
    base = {r: dequantize_hif4(quantize_to_hif4(dequantize_nvfp4(*sample[r]).float()),
                               dequantize_nvfp4(*sample[r]).float().shape).float() for r in 'qkv'}
    bm = mse(ref, gqa_attention(base['q'], base['k'], base['v'], qh, kh, dim))

    def flat(x):
        return x.transpose(0, 1).reshape(T, -1).contiguous()

    s_dep = score(bm, mse(ref, gqa_attention(dep['q'], dep['k'], dep['v'], qh, kh, dim)))
    s_q = score(bm, mse(ref, gqa_attention(flat(q1), dep['k'], dep['v'], qh, kh, dim)))
    s_qk = score(bm, mse(ref, gqa_attention(flat(q1), flat(k1), dep['v'], qh, kh, dim)))

    print('%s  group0 case%d  T=%d qh/kh/d=%d/%d/%d  sweeps=%d'
          % (Path(a.data).name, a.case, T, qh, kh, dim, a.sweeps))
    print('  部署码的 logits MSE        = %.6e    （逐张量 MSE 目标选的码）' % e0)
    print('  只重搜 K 后                = %.6e   （%+.1f%%）' % (ek, 100 * (ek / e0 - 1)))
    print('  只重搜 Q 后                = %.6e   （%+.1f%%）' % (e1, 100 * (e1 / e0 - 1)))
    print('  Q 与 K 都重搜后            = %.6e   （%+.1f%%）' % (e2, 100 * (e2 / e0 - 1)))
    print('  正向/转置目标一致性         = %.3e' % rel)
    print()
    print('  → logits-aware 编码相对逐张量 MSE 编码的可达余量 = %.1f%%'
          % (100 * (1 - e2 / e0)))
    print()
    print('  折算到**被打分的输出**：')
    print('    部署 Q/K/V            score = %+.6f  (%8.1f / 250 题)' % (s_dep, s_dep * 25000))
    print('    仅 oracle Q           score = %+.6f  (%8.1f)   Δ %+8.1f' % (s_q, s_q * 25000, (s_q - s_dep) * 25000))
    print('    oracle Q + oracle K   score = %+.6f  (%8.1f)   Δ %+8.1f' % (s_qk, s_qk * 25000, (s_qk - s_dep) * 25000))


if __name__ == '__main__':
    main()
