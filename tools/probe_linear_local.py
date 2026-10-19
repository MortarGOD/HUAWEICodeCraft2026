"""Linear-only local evaluation: mean quality and the implied 50-question online points.

quality per case = (baseline_hif4_mse - player_mse) / baseline_hif4_mse, exactly as the grader
computes it. With 50 Linear questions worth 100 points each, mean_quality * 5000 is the Linear
contribution to the online total. The user-supplied assumption is that this is currently 4450.

Usage: probe_linear_local.py --solution solution/solution_vNNN.py [--data docs/.../linear.pt]
"""
import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import load_solution, evaluate_linear  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solution', required=True)
    ap.add_argument('--data', default=str(ROOT / 'mini_sample/linear.pt'))
    ap.add_argument('--question-count', type=int, default=50)
    a = ap.parse_args()
    torch.set_num_threads(4)
    solution = load_solution(Path(a.solution).resolve())
    groups = torch.load(a.data, map_location='cpu', weights_only=False)
    scores, secs = evaluate_linear(solution, groups)
    n = len(scores)
    mean = sum(scores) / n
    print('%-34s cases=%-4d mean_quality=%.6f  implied_online_linear=%7.1f  (max %d)  wall=%.1fs'
          % (Path(a.solution).name, n, mean, mean * a.question_count * 100,
             a.question_count * 100, secs))
    print('    per-case: %s' % ' '.join('%.4f' % s for s in scores))


if __name__ == '__main__':
    main()
