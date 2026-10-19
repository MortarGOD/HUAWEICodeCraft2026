"""Scorer-side Linear error budget: how big is each side's pool, independent of our solution?

The previous attempt substituted exact weights or activations into the solution's own output and
produced nonsense (-1012 per case), because the solution's activation quantiser is weight-aware
(SmoothQuant-style coupling): replacing one side alone breaks the coordinate consistency. So
this probe never touches the solution. Everything below is computed exactly the way the grader
does it, on the standard HiF4 quantiser:

  baseline      HiF4(W) and HiF4(X)          the 0-score reference
  fixed-A       HiF4(W) and exact X          pool available purely from the activation side
  fixed-W       exact W and HiF4(X)          pool available purely from the weight side
  both exact    exact W and exact X          sanity: must be score 1.0

This reproduces the accounting discipline of the openaddr strategy doc (activation ~77% /
weight ~23%) on OUR mirror bank, using only quantities the grader defines.
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import mse, score  # noqa: E402
from hif4_quant import dequantize_nvfp4, dequantize_hif4, quantize_to_hif4  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',
                    default=str(ROOT / 'local_fitted_models_attention_feedback_v2/linear.pt'))
    ap.add_argument('--question-count', type=int, default=50)
    ap.add_argument('--groups', type=int, default=0, help='0 = all')
    a = ap.parse_args()
    torch.set_num_threads(4)
    groups = torch.load(a.data, map_location='cpu', weights_only=False)
    if a.groups:
        groups = groups[:a.groups]

    acc = {'fixedA': [], 'fixedW': [], 'both': []}
    for gi, group in enumerate(groups):
        wq, ws = group['weight']
        W = dequantize_nvfp4(wq, ws).float()
        Wb = dequantize_hif4(quantize_to_hif4(W), tuple(W.shape)).float()
        for aq, ass in group['test_activation_list']:
            X = dequantize_nvfp4(aq, ass).float()
            Xb = dequantize_hif4(quantize_to_hif4(X), tuple(X.shape)).float()
            ref = torch.matmul(X, W.transpose(-1, -2))
            bm = mse(ref, torch.matmul(Xb, Wb.transpose(-1, -2)))
            acc['fixedA'].append(score(bm, mse(ref, torch.matmul(X, Wb.transpose(-1, -2)))))
            acc['fixedW'].append(score(bm, mse(ref, torch.matmul(Xb, W.transpose(-1, -2)))))
            acc['both'].append(score(bm, 0.0))
    n = len(acc['both'])
    m = {k: sum(v) / n for k, v in acc.items()}
    print('评分端 Linear 预算（%s，cases=%d）' % (Path(a.data).name, n))
    print('  基线（两侧都是标准 HiF4）      每例 0.0000（定义）')
    print('  激活换成精确、权重保持标准    每例 %.4f  ← 激活端的池子' % m['fixedA'])
    print('  权重换成精确、激活保持标准    每例 %.4f  ← 权重端的池子' % m['fixedW'])
    print('  两侧都精确                    每例 %.4f  （恒为 1.0，自检）' % m['both'])
    print()
    tot = m['fixedA'] + m['fixedW']
    if tot > 0:
        print('  池子占比： 激活端 %.1f%%   权重端 %.1f%%'
              % (100 * m['fixedA'] / tot, 100 * m['fixedW'] / tot))
    print('  以 %d 题计：激活端最多 +%.1f 分，权重端最多 +%.1f 分'
          % (a.question_count, m['fixedA'] * a.question_count * 100,
             m['fixedW'] * a.question_count * 100))


if __name__ == '__main__':
    main()


def per_group(data, question_count):
    """Same scorer-side accounting, reported per group together with the weight shape."""
    groups = torch.load(data, map_location='cpu', weights_only=False)
    print()
    print('%-6s %14s %10s %10s %10s' % ('group', 'W shape (M,K)', 'fixedA', 'fixedW', 'ratio'))
    tot = [0.0, 0.0]
    for gi, group in enumerate(groups):
        wq, ws = group['weight']
        W = dequantize_nvfp4(wq, ws).float()
        Wb = dequantize_hif4(quantize_to_hif4(W), tuple(W.shape)).float()
        fa, fw = [], []
        for aq, ass in group['test_activation_list']:
            X = dequantize_nvfp4(aq, ass).float()
            Xb = dequantize_hif4(quantize_to_hif4(X), tuple(X.shape)).float()
            ref = torch.matmul(X, W.transpose(-1, -2))
            bm = mse(ref, torch.matmul(Xb, Wb.transpose(-1, -2)))
            fa.append(score(bm, mse(ref, torch.matmul(X, Wb.transpose(-1, -2)))))
            fw.append(score(bm, mse(ref, torch.matmul(Xb, W.transpose(-1, -2)))))
        a_, w_ = sum(fa) / len(fa), sum(fw) / len(fw)
        tot[0] += a_ / len(groups)
        tot[1] += w_ / len(groups)
        print('%-6d %14s %10.4f %10.4f %9.2f' % (gi, '%dx%d' % tuple(W.shape), a_, w_,
                                                 (a_ / w_) if w_ else float('inf')))
    print('%-6s %14s %10.4f %10.4f %9.2f' % ('mean', '', tot[0], tot[1], tot[0] / tot[1]))
