"""Paired screen for adaptive plus low-weight positional spectral V risk."""
import argparse
import contextlib
import copy
import io
import json
import time

import torch

from attention_hybrid_risk_source import hybrid_risk_source
from probe_v319_value import ROOT, load_solution
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=.05)
    parser.add_argument('--grid', type=int, default=64)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--adaptive-rank', type=int, default=16)
    parser.add_argument('--holdout', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    base = load_solution(ROOT / 'solution/solution_v353.py')
    trial = load_solution(ROOT / 'solution/solution_v353.py')
    source = hybrid_risk_source(args.alpha, args.grid, args.rank, args.adaptive_rank)
    exec(compile(source, 'attention_hybrid_risk', 'exec'), trial.__dict__)
    rows = []
    holdout = (2, 3, 4, 6, 7, 8, 9, 10, 11, 12)
    selections = (
        ('local_fitted_models_attention_feedback_v2', holdout if args.holdout else (0, 1, 5, 14, 19, 49)),
        ('local_fitted_models_covariance_v1', holdout if args.holdout else (0, 1, 24, 25, 26, 49)),
    )
    for dataset, groups in selections:
        bank = torch.load(ROOT / dataset / 'attn.pt', weights_only=False)
        for gi in groups:
            group = bank[gi]
            dims = [group[k] for k in ('q_num_heads', 'kv_num_heads', 'head_dim')]
            versions = {}
            for label, module in (('base', base), ('hybrid', trial)):
                calibration = module.hif4_calibration_attention
                start = time.perf_counter()
                state = calibration(group['calib'], *dims)
                seconds = time.perf_counter() - start
                assert not validate_attention_calibration_result(state, label)
                module.hif4_calibration_attention = lambda *_, _state=state: copy.deepcopy(_state)
                with contextlib.redirect_stdout(io.StringIO()):
                    scores, _ = evaluate_attention(module, [group])
                module.hif4_calibration_attention = calibration
                versions[label] = {'scores': scores, 'calibration_seconds': seconds}
            row = {
                'dataset': dataset,
                'group': gi,
                'versions': versions,
                'delta': sum(versions['hybrid']['scores']) - sum(versions['base']['scores']),
            }
            rows.append(row)
            print(row, flush=True)
    tag = str(args.alpha).replace('.', 'p')
    report = {'complete': True, 'alpha': args.alpha, 'grid': args.grid, 'rank': args.rank,
              'adaptive_rank': args.adaptive_rank,
              'rows': rows, 'delta': sum(row['delta'] for row in rows),
              'warning': 'Modeled activations and repeated research set; no online or full-runtime claim.'}
    suffix = '-holdout' if args.holdout else ''
    path = ROOT / f'docs/attention-hybrid-risk-a{tag}-g{args.grid}-r{args.rank}-ar{args.adaptive_rank}{suffix}.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print({'delta': report['delta'], 'path': str(path)})


if __name__ == '__main__':
    main()
