"""Which side of the Linear pool does our solution actually capture?

The earlier counterfactual failed because the solution's activation quantiser is weight-aware:
weights are quantised as W*s and activations as X/s for a per-input-channel s, so the deployed
product X_hat @ W_hat^T only cancels if both sides stay in their own coordinates. Substituting
an exact tensor on one side alone therefore produced nonsense (-1012 per case).

Fix: recover s from the solution's own weight parameters by per-channel least squares against
the true weight,

    s_j = <W_hat[:,j], W[:,j]> / <W[:,j], W[:,j]>,

then rebuild the two effective tensors in the ORIGINAL coordinates,
    X_hat = activation_params * s      W_hat = weight_params / s,
and score the four outputs the grader defines. s is only a diagnostic reconstruction: every
number reported is still an MSE between real attention/linear outputs and the NVFP4 reference,
computed the same way the grader does it.
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import load_solution, mse, score  # noqa: E402
from hif4_quant import dequantize_nvfp4, dequantize_hif4, quantize_to_hif4  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solution', required=True)
    ap.add_argument('--data',
                    default=str(ROOT / 'local_fitted_models_attention_feedback_v2/linear.pt'))
    ap.add_argument('--question-count', type=int, default=50)
    a = ap.parse_args()
    torch.set_num_threads(4)
    solution = load_solution(Path(a.solution).resolve())
    groups = torch.load(a.data, map_location='cpu', weights_only=False)

    acc = {k: [] for k in ('base', 'deployed', 'perfectW', 'perfectA')}
    for group in groups:
        wq, ws = group['weight']
        W = dequantize_nvfp4(wq, ws).float()
        Wb = dequantize_hif4(quantize_to_hif4(W), tuple(W.shape)).float()
        res = solution.hif4_calibration_and_quantize_weight(
            wq, ws, group['calib_activation_list'])
        Wh = dequantize_hif4(res['weight_params'], tuple(W.shape)).float()
        # per-input-channel scaling the solution folded in, via least squares
        denom = (W * W).sum(0).clamp_min(1e-30)
        s = (Wh * W).sum(0) / denom
        s = torch.where(s.abs() < 1e-6, torch.ones_like(s), s)
        What = Wh / s
        for aq, ass in group['test_activation_list']:
            X = dequantize_nvfp4(aq, ass).float()
            Xb = dequantize_hif4(quantize_to_hif4(X), tuple(X.shape)).float()
            ref = torch.matmul(X, W.transpose(-1, -2))
            bm = mse(ref, torch.matmul(Xb, Wb.transpose(-1, -2)))
            Xh = dequantize_hif4(
                solution.hif4_dynamic_quantize_activation(aq, ass, res['activation_state']),
                tuple(X.shape)).float() * s
            acc['base'].append(bm)
            acc['deployed'].append(score(bm, mse(ref, torch.matmul(Xh, What.transpose(-1, -2)))))
            acc['perfectW'].append(score(bm, mse(ref, torch.matmul(Xh, W.transpose(-1, -2)))))
            acc['perfectA'].append(score(bm, mse(ref, torch.matmul(X, What.transpose(-1, -2)))))
    n = len(acc['deployed'])
    m = {k: sum(v) / n for k, v in acc.items() if k != 'base'}
    # scorer-side totals for reference
    print('%s  cases=%d' % (Path(a.solution).name, n))
    print('  部署态                       每例 %.4f' % m['deployed'])
    print('  用我们的激活 + 精确权重      每例 %.4f   ← 只剩权重端误差' % m['perfectW'])
    print('  用精确激活 + 我们的权重      每例 %.4f   ← 只剩激活端误差' % m['perfectA'])
    print()
    print('  可加池子参照（评分端）：激活 +0.5367 / 权重 +0.4640 / 合计 +1.0007')
    print('  我们的部署态 %.4f 占总池 %.1f%%' % (m['deployed'], 100 * m['deployed'] / 1.0007))
    still = 1.0007 - m['deployed']
    print('  剩余池子 %.4f/case = %d 题合计 %.1f 分' % (still, a.question_count,
                                                still * a.question_count * 100))


if __name__ == '__main__':
    main()
