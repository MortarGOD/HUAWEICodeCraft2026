"""Synthetic multi-distribution bank, ported from the openaddr-hif4-contest repo.

Source: docs/papers/hif4-public-references/openaddr-hif4-contest/dev/synth.py (VERSION 45).

Purpose here is NEGATIVE screening, exactly as that repo uses it: the generated tensors are
iid Gaussian times per-channel gains, so they carry no repeatable channel structure. A
mechanism that gains on the real mirror bank but collapses here is fitting the specific
calibration sample rather than a structural property, and that repo's dead-end list records
four theorem-level cases where such fits never transferred to the judge
("样本拟合变换永不转移，只有结构不可知机制能转移").

Consequence for this project: a synthetic result is NEVER evidence for an online claim
(the local proxy was falsified by the v501/v521 receipts). It is only a rejection filter.

Carrier format matches the contest: NVFP4 = E2M1 grid {0,0.5,1,1.5,2,3,4,6}, per-16-element
block scale = absmax/6 stored in bf16, both returned as (carrier, scale) bf16 pairs.

Usage:
    python tools/synth_bank.py --out-dir /tmp/synth_bank --groups 8 --linear-shapes 512x2048 ...
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_THRESHOLDS = (E2M1_GRID[1:] + E2M1_GRID[:-1]) / 2.0


def quant_pair(x: torch.Tensor, blk: int = 16):
    """Return (carrier, scale) bf16 carrier pair for a 2-D tensor, NVFP4-simulated."""
    rows = x.shape[0]
    xb = x.float().reshape(-1, blk)
    amax = xb.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    scale = ((amax / 6.0).to(torch.bfloat16).float()).clamp_min(1e-30)
    q = xb / scale
    idx = torch.bucketize(q.abs().contiguous(), _THRESHOLDS)
    carrier = (torch.sign(q) * E2M1_GRID[idx]).reshape(rows, -1).to(torch.bfloat16).contiguous()
    return carrier, scale.reshape(rows, -1).to(torch.bfloat16).contiguous()


def chan_gains(c: int, spread: float, gen: torch.Generator) -> torch.Tensor:
    return torch.exp((torch.rand(c, generator=gen) - 0.5) * 2 * math.log(10.0) * spread)


def _raw(rows: int, cols: int, spread: float, outlier_p: float, gen: torch.Generator):
    x = (torch.randn(rows, 1, generator=gen)
         * chan_gains(cols, spread, gen).unsqueeze(0)
         * torch.randn(rows, cols, generator=gen))
    if outlier_p > 0:
        mask = torch.rand(rows, cols, generator=gen) < outlier_p
        x = x + mask.float() * torch.randn(rows, cols, generator=gen) * x.abs().amax() * 3
    return x


def make_linear_group(seed: int, m: int, k: int, tokens, spread=0.5, outlier_p=0.0,
                      w_spread=0.3):
    gen = torch.Generator().manual_seed(seed)
    w = (torch.randn(m, 1, generator=gen)
         * chan_gains(k, w_spread, gen).unsqueeze(0)
         * torch.randn(m, k, generator=gen)) * 0.05
    wc, ws = quant_pair(w)
    return {
        'weight': (wc, ws),
        'calib_activation_list': [quant_pair(_raw(t, k, spread, outlier_p, gen)) for t in tokens],
        'test_activation_list': [quant_pair(_raw(t, k, spread, outlier_p, gen)) for t in tokens],
    }


def make_attn_group(seed: int, qh: int, kvh: int, dh: int, seqlens, spread=0.4,
                    outlier_p=0.0):
    gen = torch.Generator().manual_seed(seed)
    qc, kc = qh * dh, kvh * dh

    def sample(t):
        return {
            'q': quant_pair(_raw(t, qc, spread, outlier_p, gen)),
            'k': quant_pair(_raw(t, kc, spread * 0.7, outlier_p, gen)),
            'v': quant_pair(_raw(t, kc, spread * 0.5, outlier_p, gen)),
        }

    return {'q_num_heads': qh, 'kv_num_heads': kvh, 'head_dim': dh,
            'calib': [sample(t) for t in seqlens], 'test': [sample(t) for t in seqlens]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--groups', type=int, default=8)
    ap.add_argument('--seed', type=int, default=20260910)
    ap.add_argument('--linear-shapes', default='2048x512,512x2048,1024x2048,2048x1024')
    ap.add_argument('--tools', default='linear,attention')
    ap.add_argument('--outlier-p', type=float, default=0.0)
    ap.add_argument('--spread', type=float, default=0.5)
    ap.add_argument('--tokens', default='128,512')
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shapes = [tuple(int(v) for v in s.split('x')) for s in a.linear_shapes.split(',')]
    tokens = [int(v) for v in a.tokens.split(',')]
    want = set(a.tools.split(','))
    if 'linear' in want:
        groups = []
        for i in range(a.groups):
            m, k = shapes[i % len(shapes)]
            groups.append(make_linear_group(a.seed + i, m, k, tokens,
                                            spread=a.spread, outlier_p=a.outlier_p))
        torch.save(groups, out / 'linear.pt')
        print('wrote %s/linear.pt  groups=%d shapes=%s tokens=%s outlier_p=%s'
              % (out, len(groups), shapes, tokens, a.outlier_p))
    if 'attention' in want:
        combos = [(16, 2, 256), (8, 2, 256), (40, 8, 128), (32, 8, 128)]
        groups = []
        for i in range(a.groups):
            qh, kh, dh = combos[i % len(combos)]
            groups.append(make_attn_group(a.seed + 1000 + i, qh, kh, dh, tokens,
                                          spread=a.spread * 0.8, outlier_p=a.outlier_p))
        torch.save(groups, out / 'attn.pt')
        print('wrote %s/attn.pt  groups=%d combos=%s tokens=%s outlier_p=%s'
              % (out, len(groups), combos, tokens, a.outlier_p))


if __name__ == '__main__':
    main()
