"""Attention-only local evaluation on the structural-v3 mirror bank (250 cases).

Prints mean quality and the implied 250-question online attention points, plus wall time.
Used only to RANK probes: the local proxy was falsified by the v501/v521 receipts, so a
positive local delta is not evidence of an online gain and a zero is not disproof.
"""
import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import load_solution, evaluate_attention  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solution', required=True)
    ap.add_argument('--data', default=str(ROOT / 'local_fitted_models_attention_feedback_v2/attn.pt'))
    ap.add_argument('--question-count', type=int, default=250)
    ap.add_argument('--groups', type=int, default=0, help='0 = all')
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('EVAL_THREADS', '4')))
    solution = load_solution(Path(a.solution).resolve())
    groups = torch.load(a.data, map_location='cpu', weights_only=False)
    if a.groups:
        groups = groups[:a.groups]
    scores, secs = evaluate_attention(solution, groups)
    n = len(scores)
    mean = sum(scores) / n
    print('%-30s cases=%-4d mean_quality=%.6f implied_online_attn=%8.1f (max %d) wall=%.1fs'
          % (Path(a.solution).name, n, mean, mean * a.question_count * 100,
             a.question_count * 100, secs))


if __name__ == '__main__':
    main()
