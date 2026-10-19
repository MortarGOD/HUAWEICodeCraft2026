"""Does the code-shaping gain survive when H is only a proxy?

The greedy result (tools/probe_v_code_shaping.py) uses the true H = sum_h P_h^T P_h of the very
sample being shaped, which the V dynamic interface cannot see: the V call receives only V, and
cross-role state is forbidden. So any implementable mechanism must use a surrogate.

Two surrogates are tested here, both measured against the TRUE H of the shaped sample:
  (a) H taken from a DIFFERENT sample of the same length (cross-sample transfer),
  (b) H taken from the calibration sample of the same length,
  (c) fixed kernels that need no P at all: flat (causal mean), and exponentially decaying
      with several rates.

If (a)/(b) collapse, no proxy can work and the direction is dead. If (c) is weak but (a)/(b)
hold, the gain is real but only usable when a matching-length calibration H exists.
"""
import copy
import sys
from pathlib import Path

import torch

ROOT = Path('/Users/ee/Code/2026-huawei-alogorithm')
sys.path.insert(0, str(ROOT))
from hif4_quant import dequantize_nvfp4, dequantize_hif4  # noqa: E402
from tools.test_v422 import load  # noqa: E402

torch.set_num_threads(4)
torch.manual_seed(20260910)
VERSION = 441
DS = ROOT / 'docs/papers/hif4-public-references/test/datasets/public_quantizer/attn.pt'
SWEEPS = 10
RATES = (0.0, 0.01, 0.03, 0.1, 0.3)


def softmax_maps(q, k, qh, kh, d):
    T = q.shape[0]
    qf = q.reshape(T, qh, d).transpose(0, 1)
    kf = k.reshape(T, kh, d).transpose(0, 1).repeat_interleave(qh // kh, 0)
    return torch.softmax(qf @ kf.transpose(1, 2) / (d ** 0.5), -1)


def group_H(P, gi, rep):
    Hg = P[gi * rep:(gi + 1) * rep]
    return (Hg.transpose(1, 2) @ Hg).sum(0).double()


def shape(D, Htrue, Huse, step, sweeps=SWEEPS):
    """Greedy coordinate descent using Huse for decisions; Htrue is only for scoring."""
    Dg = D.clone()
    T, d = D.shape
    for _ in range(sweeps):
        HD = Huse @ Dg
        any_move = False
        for t in range(T):
            deltas = torch.stack([-step[t], torch.zeros(d, dtype=torch.float64), step[t]])
            cand = 2.0 * deltas * HD[t][None, :] + (deltas ** 2) * Huse[t, t]
            best = cand.argmin(0)
            shift = deltas.gather(0, best[None, :]).squeeze(0)
            if bool((shift != 0).any()):
                any_move = True
                Dg[t] = Dg[t] + shift
                HD = HD + Huse[:, t][:, None] * shift[None, :]
        if not any_move:
            break
    return float((Dg * (Htrue @ Dg)).sum())


@torch.inference_mode()
def main():
    g = torch.load(DS, weights_only=True)[0]
    qh, kh, d = g['q_num_heads'], g['kv_num_heads'], g['head_dim']
    cal = load(VERSION, 'calib_q_0').hif4_calibration_attention(copy.deepcopy(g['calib']), qh, kh, d)
    mods = {r: load(VERSION, 'dyn_%s_0' % r) for r in 'qkv'}
    rep = qh // kh
    b = (kh * d) // 64

    def runtime(sample, states=None):
        raw = {r: dequantize_nvfp4(*sample[r]).float() for r in 'qkv'}
        dq, par = {}, {}
        for role in 'qkv':
            st = copy.deepcopy(cal[role + '_state'])
            fn = getattr(mods[role], 'hif4_dynamic_quantize_' + role)
            par[role] = fn(*sample[role], qh if role == 'q' else kh, d, st)
            dq[role] = dequantize_hif4(par[role], raw[role].shape)
        return raw, dq, par

    tests = {i: runtime(s) for i, s in enumerate(g['test'])}
    calib = {i: runtime(s) for i, s in enumerate(g['calib'])}
    print('%-6s %-26s %12s %9s' % ('T', 'surrogate H', 'shaped f', 'gain%'))
    for i, (raw, dq, par) in tests.items():
        T = raw['v'].shape[0]
        if T < 512:
            continue
        Ptrue = softmax_maps(dq['q'], dq['k'], qh, kh, d)
        # score with group 0 only for clarity
        gi = 0
        Htrue = group_H(Ptrue, gi, rep)
        unit = (par['v']['scale_factor'].double() * par['v']['scale_lv2'].double()
                * par['v']['scale_lv3'].double()).expand(T, b, 8, 2, 4).reshape(T, kh * d)
        unit = unit.reshape(T, kh, d)[:, gi, :].double()
        D = (dq['v'] - raw['v']).reshape(T, kh, d)[:, gi, :].double()
        base = float((D * (Htrue @ D)).sum())
        step = 0.25 * unit
        surrogates = {'true H (upper bound)': Htrue}
        for j, (r2, d2, p2) in tests.items():
            if j == i or d2['v'].shape[0] != T:
                continue
            surrogates['other test sample H'] = group_H(softmax_maps(d2['q'], d2['k'], qh, kh, d), gi, rep)
        for j, (r2, d2, p2) in calib.items():
            if d2['v'].shape[0] != T:
                continue
            surrogates['calib sample H'] = group_H(softmax_maps(d2['q'], d2['k'], qh, kh, d), gi, rep)
        for rate in RATES:
            idx = torch.arange(T).double()
            K = torch.exp(-rate * (idx[:, None] - idx[None, :]).abs())
            K = K / K.sum(-1, keepdim=True)
            surrogates['kernel decay=%.2f' % rate] = K
        for name, Huse in surrogates.items():
            f = shape(D, Htrue, Huse, step)
            print('%-6d %-26s %12.5e %8.1f%%' % (T, name, f, 100 * (1 - f / base)))
        print()


if __name__ == '__main__':
    main()
