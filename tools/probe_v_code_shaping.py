"""Can deliberate non-nearest rounding reduce the H-weighted V error?

The deployed V coder always rounds to the nearest S1P2 code, which is optimal for the
element-wise metric but not for the deployed one: the objective is

    f(D) = sum_j D_j^T H D_j,     H = sum_{h in group} P_h^T P_h,   D = V_hat - V.

Nothing in the format forces the nearest code. Shifting one element by one S1P2 step
(spacing 0.25 * unit) changes f by  2*d*(H D_j)[t] + d^2 * H[t,t],  so a discrete greedy
coordinate descent over {code-1, code, code+1} finds an achievable improvement. That is an
error-shaping step in the GPTQ family (compensating already-fixed errors by biasing later
rounding decisions).

This measures the achievable reduction and, separately, how many elements are moved, because
a mechanism that rewrites most codes is expensive on the dynamic path.
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
SWEEPS = 12
MAX_T = 1024


def softmax_maps(q, k, qh, kh, d):
    T = q.shape[0]
    qf = q.reshape(T, qh, d).transpose(0, 1)
    kf = k.reshape(T, kh, d).transpose(0, 1).repeat_interleave(qh // kh, 0)
    return torch.softmax(qf @ kf.transpose(1, 2) / (d ** 0.5), -1)


@torch.inference_mode()
def main():
    g = torch.load(DS, weights_only=True)[0]
    qh, kh, d = g['q_num_heads'], g['kv_num_heads'], g['head_dim']
    cal = load(VERSION, 'calib_q_0').hif4_calibration_attention(copy.deepcopy(g['calib']), qh, kh, d)
    mods = {r: load(VERSION, 'dyn_%s_0' % r) for r in 'qkv'}
    print('%-5s %12s %12s %8s %8s %8s' % ('toks', 'deployed f', 'greedy f', 'gain%',
                                           'moved%', 'max|move|'))
    tot = [0.0, 0.0]
    for s in g['test']:
        raw = {r: dequantize_nvfp4(*s[r]).float() for r in 'qkv'}
        dq, par = {}, {}
        for role in 'qkv':
            st = copy.deepcopy(cal[role + '_state'])
            fn = getattr(mods[role], 'hif4_dynamic_quantize_' + role)
            par[role] = fn(*s[role], qh if role == 'q' else kh, d, st)
            dq[role] = dequantize_hif4(par[role], raw[role].shape)
        P = softmax_maps(dq['q'], dq['k'], qh, kh, d)
        T = raw['v'].shape[0]
        if T > MAX_T:
            continue
        rep = qh // kh
        cur = 0.0
        low = 0.0
        moved = 0
        totel = 0
        maxmove = 0.0
        for gi in range(kh):
            Hg = P[gi * rep:(gi + 1) * rep]
            H = (Hg.transpose(1, 2) @ Hg).sum(0).double()          # (T,T)
            V = raw['v'].reshape(T, kh, d)[:, gi, :].double()
            b = (kh * d) // 64
            unit_full = (par['v']['scale_factor'].double() * par['v']['scale_lv2'].double()
                         * par['v']['scale_lv3'].double()).expand(T, b, 8, 2, 4).reshape(T, kh * d)
            unit = unit_full.reshape(T, kh, d)[:, gi, :].double()
            D = (dq['v'] - raw['v']).reshape(T, kh, d)[:, gi, :].double()
            cur += float((D * (H @ D)).sum())
            step = 0.25 * unit                                   # one S1P2 code, per element
            Dg = D.clone()
            for _ in range(SWEEPS):
                HD = H @ Dg
                any_move = False
                for t in range(T):
                    deltas = torch.stack([-step[t], torch.zeros(d, dtype=torch.float64),
                                          step[t]])
                    # f is quadratic, so moving D[t,j] by delta changes it by
                    #   2*delta*(HD)[t,j] + delta^2*H[t,t]
                    # (delta = 0 is a candidate, so the chosen move never increases f)
                    cand = (2.0 * deltas * HD[t][None, :]
                            + (deltas ** 2) * H[t, t])
                    best = cand.argmin(0)
                    shift = deltas.gather(0, best[None, :]).squeeze(0)
                    if bool((shift != 0).any()):
                        any_move = True
                        Dg[t] = Dg[t] + shift
                        HD = HD + H[:, t][:, None] * shift[None, :]     # incremental update
                if not any_move:
                    break
            moved += int((Dg != D).sum())
            totel += D.numel()
            maxmove = max(maxmove, float((Dg - D).abs().max()))
            low += float((Dg * (H @ Dg)).sum())
        n_elem = qh * T * d
        print('%-5d %12.4e %12.4e %7.1f%% %7.1f%% %8.3e'
              % (T, cur / n_elem, low / n_elem, 100 * (1 - low / cur),
                 100 * moved / max(totel, 1), maxmove))
        tot[0] += cur / n_elem
        tot[1] += low / n_elem
    print()
    print('合计: 部署 = %.4e  贪心整形后 = %.4e  可达降低 = %.1f%%'
          % (tot[0], tot[1], 100 * (1 - tot[1] / tot[0])))
    print('对照：23000 档需要总 player MSE 降 12.0%%（若全部由 V 承担则需 V 误差降 15.3%%）')


if __name__ == '__main__':
    main()
