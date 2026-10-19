"""Paired screen for the fixed low-rank calibration attention Gram."""
import argparse
import contextlib
import copy
import io
import json
import time

import torch

from attention_spectral_risk_source import spectral_risk_source
from probe_v319_value import ROOT, load_solution
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', type=int, default=128)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--cross-guard', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    base = load_solution(ROOT / 'solution/solution_v353.py')
    trial = load_solution(ROOT / 'solution/solution_v353.py')
    source = spectral_risk_source(args.grid, args.rank, args.cross_guard)
    exec(compile(source, 'attention_spectral_risk', 'exec'), trial.__dict__)
    rows = []
    for dataset, groups in (
        ('local_fitted_models_attention_feedback_v2', (0, 1, 5, 14, 19, 49)),
        ('local_fitted_models_covariance_v1', (0, 1, 24, 25, 26, 49)),
    ):
        bank = torch.load(ROOT / dataset / 'attn.pt', weights_only=False)
        for gi in groups:
            group = bank[gi]
            dims = [group[k] for k in ('q_num_heads', 'kv_num_heads', 'head_dim')]
            versions = {}
            for label, module in (('base', base), ('spectral', trial)):
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
                'delta': sum(versions['spectral']['scores']) - sum(versions['base']['scores']),
            }
            rows.append(row)
            print(row, flush=True)
    report = {
        'complete': True,
        'grid': args.grid,
        'rank': args.rank,
        'cross_guard': args.cross_guard,
        'rows': rows,
        'delta': sum(row['delta'] for row in rows),
        'warning': 'Modeled activations and repeated research set; no online or full-runtime claim.',
    }
    suffix = '-cross' if args.cross_guard else ''
    path = ROOT / f'docs/attention-spectral-risk-g{args.grid}-r{args.rank}{suffix}.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print({'delta': report['delta'], 'path': str(path)})


if __name__ == '__main__':
    main()
