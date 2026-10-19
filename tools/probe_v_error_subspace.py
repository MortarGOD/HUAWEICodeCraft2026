"""Is the V error budget hiding in the token-mean subspace?

The deployed objective for V is, per KV group g,

    ||Delta_out||^2 = sum_j  dV_g[:,j]^T H_g dV_g[:,j],     H_g = sum_{h in g} P_h^T P_h

where P_h is the softmax map of query head h. The parent's V path makes its scale/micro-
exponent decisions per 64-element block, and a block lives inside a single token, so every
decision is blind to the off-diagonal (cross-token) part of H_g.

If H_g is dominated by its diagonal, per-token decisions are near-optimal and there is
nothing to gain. If H_g is closer to rank one (all-ones outer product), then only the
per-channel MEAN error matters and large per-token errors that cancel across tokens are
free -- which is exactly what error shaping / GPTQ over the token axis can exploit.

This script measures, on public_quantizer with the v441 path:
  * the true V-attributable output error (validating the H_g machinery),
  * the plain element-wise error,
  * the split of the H-weighted error into the per-channel mean part and the rest,
  * the token-mean fraction of H_g (H diag sum vs total sum) as a rank-one proxy,
  * how much the V-attributable error would drop if the per-channel mean error were removed.
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


def softmax_maps(q, k, qh, kh, d):
    """P: (qh, T, T) built exactly like the scorer's attention."""
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
    print('%-5s %11s %11s %11s %11s %9s %9s %9s'
          % ('toks', 'H-weighted', 'plain/2', 'mean part', 'resid part', 'HdiagFrac', 'trueOutMSE', 'HoutMSE'))
    tot = {k: 0.0 for k in ('hw', 'pl', 'mn', 'rs', 'tm', 'dh', 'td', 'dg')}
    for s in g['test']:
        raw = {r: dequantize_nvfp4(*s[r]).float() for r in 'qkv'}
        dq = {}
        for role in 'qkv':
            st = copy.deepcopy(cal[role + '_state'])
            fn = getattr(mods[role], 'hif4_dynamic_quantize_' + role)
            dq[role] = dequantize_hif4(fn(*s[role], qh if role == 'q' else kh, d, st), raw[role].shape)
        P = softmax_maps(dq['q'], dq['k'], qh, kh, d)          # (qh,T,T)
        T = raw['v'].shape[0]
        dV = (dq['v'] - raw['v']).reshape(T, kh, d)
        hw = 0.0          # sum_j dV^T H dV, H-weighted
        pl = 0.0
        mn = 0.0
        rs = 0.0
        dg = 0.0          # diagonal-only part of H
        diag_frac = 0.0
        rep = qh // kh
        for gi in range(kh):
            Hg = P[gi * rep:(gi + 1) * rep]
            H = (Hg.transpose(1, 2) @ Hg).sum(0).double()      # (T,T), summed over the group
            D = dV[:, gi, :].double()                          # (T,d)
            hw += float((D * (H @ D)).sum())
            pl += float(D.square().sum()) / 2.0                # half, so scale matches H for uniform P
            dg += float((H.diagonal()[:, None] * D.square()).sum())
            m = D.mean(0, keepdim=True)
            Dm = m.expand_as(D)
            Dr = D - Dm
            mn += float((Dm * (H @ Dm)).sum())
            rs += float((Dr * (H @ Dr)).sum())
            diag_frac += float(H.diagonal().sum()) / float(H.sum().clamp_min(1e-30))
        diag_frac /= kh
        Vrep = dq['v'].reshape(T, kh, d).repeat_interleave(rep, 1)
        Vraw = raw['v'].reshape(T, kh, d).repeat_interleave(rep, 1)
        out_v = torch.einsum('hij,jhd->ihd', P, Vrep).reshape(-1)
        ref_v = torch.einsum('hij,jhd->ihd', P, Vraw).reshape(-1)
        true_mse = float((out_v - ref_v).square().mean())
        h_mse = hw / (qh * T * d)
        print('%-5d %11.4e %11.4e %11.4e %11.4e %9.4f %11.4e %11.4e'
              % (T, hw / (qh * T * d), pl / (qh * T * d), mn / (qh * T * d), rs / (qh * T * d),
                 diag_frac, true_mse, h_mse))
        for k, v in (('hw', hw / (qh * T * d)), ('pl', pl / (qh * T * d)), ('mn', mn / (qh * T * d)),
                     ('rs', rs / (qh * T * d)), ('tm', true_mse), ('dh', h_mse), ('td', diag_frac),
                     ('dg', dg / (qh * T * d))):
            tot[k] += v / len(g['test'])
    print()
    print('平均 Hdiag 占比（1.0 = 完全对角，1/T = 完全 rank-one 均匀） = %.4f，T 平均 = %d (1/T≈%.4f)'
          % (tot['td'], sum(dequantize_nvfp4(*s['v']).shape[0] for s in g['test']) / 5,
             1.0 / (sum(dequantize_nvfp4(*s['v']).shape[0] for s in g['test']) / 5)))
    print('V 归因输出 MSE: 直接算 = %.4e   H 公式 = %.4e   （应相等，验证 H 机制）' % (tot['tm'], tot['dh']))
    print('其中 逐通道均值部分 = %.4e (%.1f%%)   残差部分 = %.4e (%.1f%%)'
          % (tot['mn'], 100 * tot['mn'] / tot['hw'], tot['rs'], 100 * tot['rs'] / tot['hw']))
    print('逐元素误差(½) = %.4e ；H 加权 = %.4e ；比值 = %.3f'
          % (tot['pl'], tot['hw'], tot['hw'] / tot['pl']))
    print('H 的纯对角部分 = %.4e，占 H 加权 %.1f%%  -> 非对角（可被 token 方向 GPTQ 补偿）最多 %.1f%%'
          % (tot['dg'], 100 * tot['dg'] / tot['hw'], 100 * (1 - tot['dg'] / tot['hw'])))
    print()
    print('若把每通道的 token 均值误差清零，V 归因误差期望降到 %.4e（降 %.1f%%）'
          % (tot['rs'], 100 * (1 - tot['rs'] / tot['hw'])))


if __name__ == '__main__':
    main()
