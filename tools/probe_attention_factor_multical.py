"""Paired screen for fitting factor risk on all pre-holdout calibrations."""
import argparse
import contextlib
import copy
import io
import json
import time

import torch

from attention_factor_multical_source import multical_factor_source
from probe_v319_value import ROOT, load_solution
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--holdout', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    base = load_solution(ROOT / 'solution/solution_v353.py')
    trial = load_solution(ROOT / 'solution/solution_v353.py')
    exec(compile(multical_factor_source(), 'attention_factor_multical', 'exec'), trial.__dict__)
    held = (2, 3, 4, 6, 7, 8, 9, 10, 11, 12)
    selections = (
        ('local_fitted_models_attention_feedback_v2', held if args.holdout else (0, 1, 5, 14, 19, 49)),
        ('local_fitted_models_covariance_v1', held if args.holdout else (0, 1, 24, 25, 26, 49)),
    )
    rows = []
    for dataset, groups in selections:
        bank = torch.load(ROOT / dataset / 'attn.pt', weights_only=False)
        for group_index in groups:
            group = bank[group_index]
            dims = [group[key] for key in ('q_num_heads', 'kv_num_heads', 'head_dim')]
            versions = {}
            for label, module in (('base', base), ('multical', trial)):
                calibration = module.hif4_calibration_attention
                start = time.perf_counter()
                state = calibration(group['calib'], *dims)
                elapsed = time.perf_counter() - start
                assert not validate_attention_calibration_result(state, label)
                module.hif4_calibration_attention = lambda *_, _state=state: copy.deepcopy(_state)
                with contextlib.redirect_stdout(io.StringIO()):
                    scores, _ = evaluate_attention(module, [group])
                module.hif4_calibration_attention = calibration
                versions[label] = {'scores': scores, 'calibration_seconds': elapsed}
            row = {'dataset': dataset, 'group': group_index, 'versions': versions,
                   'delta': sum(versions['multical']['scores']) - sum(versions['base']['scores'])}
            rows.append(row)
            print(row, flush=True)
    suffix = '-holdout' if args.holdout else ''
    report = {'complete': True, 'rows': rows, 'delta': sum(row['delta'] for row in rows),
              'warning': 'Modeled activations and repeated research set; no online or full-runtime claim.'}
    path = ROOT / f'docs/attention-factor-multical{suffix}.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print({'delta': report['delta'], 'path': str(path)})


if __name__ == '__main__':
    main()
