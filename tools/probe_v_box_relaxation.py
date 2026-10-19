"""How much of the V error is removable by shaping the rounding residual along tokens?

The deployed objective for one KV group is  sum_j D_j^T H D_j  with H = sum_{h in group} P_h^T P_h
and D = V_hat - V. Each element's residual is bounded by half the local S1P2 step, so a
relaxation of "what could a smarter per-block choice achieve" is the box-constrained problem

    minimise  sum_j D_j^T H D_j     s.t.  |D[t,j]| <= delta[t,j]

solved here by coordinate descent (exact per-coordinate minimisation + clipping), started from
the deployed residual. This is a LOWER bound on the achievable error, so if the bound is close
to the current error the whole direction is dead regardless of implementation cleverness.

delta[t,j] = unit(t,j)/8, where unit = scale_factor * scale_lv2 * scale_lv3 (S1P2 spacing 0.25).
"""
import copy
import sys
from pathlib import Path

import torch

ROOT = Path('/Users/ee/Code/2026-huawei-alogorithm')
sys.path.insert(0, str(ROOT))
from hif4_quant import dequantize_nvfp4, dequantize_hif4  # noqa: E402
from tools.test_v422 import load  # noqa: E402
from evaluate_local import gqa_attention, mse, score  # noqa: E402

torch.set_num_threads(4)
torch.manual_seed(20260910)
VERSION = 441
DS = ROOT / 'docs/papers/hif4-public-references/test/datasets/public_quantizer/attn.pt'
SWEEPS = 30


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
    print('%-5s %12s %12s %9s %9s   %s' % ('toks', 'deployed H-err', 'box-relax min',
                                           'reachable', 'gain%', '（relative to plain MSE）'))
    tot = [0.0, 0.0, 0.0]
    for s in g['test']:
        raw = {r: dequantize_nvfp4(*s[r]).float() for r in 'qkv'}
        dq = {}
        par = {}
        for role in 'qkv':
            st = copy.deepcopy(cal[role + '_state'])
            fn = getattr(mods[role], 'hif4_dynamic_quantize_' + role)
            par[role] = fn(*s[role], qh if role == 'q' else kh, d, st)
            dq[role] = dequantize_hif4(par[role], raw[role].shape)
        P = softmax_maps(dq['q'], dq['k'], qh, kh, d)
        T = raw['v'].shape[0]
        rep = qh // kh
        cur = 0.0
        low = 0.0
        for gi in range(kh):
            Hg = P[gi * rep:(gi + 1) * rep]
            H = (Hg.transpose(1, 2) @ Hg).sum(0).double()          # (T,T)
            V = raw['v'].reshape(T, kh, d)[:, gi, :].double()
            D = (dq['v'] - raw['v']).reshape(T, kh, d)[:, gi, :].double()
            # per-element box: half of 0.25 * unit; unit = scale_factor * scale_lv2 * scale_lv3
            sf = par['v']['scale_factor'].double()                 # (T,b,1,1,1)
            lv2 = par['v']['scale_lv2'].double()                   # (T,b,8,1,1)
            lv3 = par['v']['scale_lv3'].double()                   # (T,b,8,2,1)
            b = (kh * d) // 64
            unit_full = (sf * lv2 * lv3).expand(T, b, 8, 2, 4).reshape(T, kh * d)
            unit_kv = unit_full.reshape(T, kh, d)[:, gi, :].double()
            delta = unit_kv / 8.0
            f_cur = float((D * (H @ D)).sum())
            cur += f_cur
            # coordinate descent on the box-constrained relaxation
            Dc = D.clone()
            for _ in range(SWEEPS):
                HD = H @ Dc
                diag = H.diagonal()[:, None]
                off = HD - diag * Dc
                new = torch.clamp(-off / diag.clamp_min(1e-30), -delta, delta)
                if bool((new - Dc).abs().max() < 1e-12):
                    Dc = new
                    break
                Dc = new
            low += float((Dc * (H @ Dc)).sum())
        n_elem = qh * T * d
        print('%-5d %12.4e %12.4e %8.1f%% %8.1f%%   plain=%.4e'
              % (T, cur / n_elem, low / n_elem, 100 * low / cur, 100 * (1 - low / cur),
                 (dq['v'] - raw['v']).square().mean().item()))
        tot[0] += cur / n_elem / len(g['test'])
        tot[1] += low / n_elem / len(g['test'])
    print()
    print('平均: 部署 H 误差 = %.4e ；盒约束松弛下界 = %.4e ；可达上限 = %.1f%%'
          % (tot[0], tot[1], 100 * (1 - tot[1] / tot[0])))
    print()
    print('对照：总 player MSE 3.7990e-04，23000 档需要总 MSE 降到 3.3428e-04（=-12.0%%）')


if __name__ == '__main__':
    main()
