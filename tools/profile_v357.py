"""Sequential same-input v353/v357 calibration and dynamic timing."""
import argparse
import contextlib
import io
import json
import time

import torch

from probe_v319_value import ROOT, load_solution
from evaluate_local import evaluate_attention


def run(module, groups):
    timing = {'calibration': 0., 'q': 0., 'k': 0., 'v': 0.}
    depth = 0
    original_calibration = module.hif4_calibration_attention

    def calibration(*args):
        nonlocal depth
        depth += 1
        start = time.perf_counter()
        try:
            return original_calibration(*args)
        finally:
            timing['calibration'] += time.perf_counter() - start
            depth -= 1

    module.hif4_calibration_attention = calibration
    for role in 'qkv':
        name = 'hif4_dynamic_quantize_' + role
        original = getattr(module, name)

        def dynamic(*args, _role=role, _original=original):
            start = time.perf_counter()
            try:
                return _original(*args)
            finally:
                if not depth:
                    timing[_role] += time.perf_counter() - start

        setattr(module, name, dynamic)
    with contextlib.redirect_stdout(io.StringIO()):
        scores, elapsed = evaluate_attention(module, groups)
    return {'timing': timing, 'score': sum(scores), 'evaluation_seconds': elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    bank = torch.load(ROOT / 'local_fitted_models_attention_feedback_v2/attn.pt', weights_only=False)
    ids = (0, 1, 5, 14, 19, 49)
    groups = [bank[index] for index in ids]
    report = {'groups': ids, 'results': {},
              'warning': 'Local 4-thread single pass; not an online 96-core or whole-suite guarantee.'}
    order = (357, 353) if args.reverse else (353, 357)
    for version in order:
        report['results'][str(version)] = run(load_solution(ROOT / f'solution/solution_v{version}.py'), groups)
        print(version, report['results'][str(version)], flush=True)
    suffix = '-reverse' if args.reverse else ''
    (ROOT / f'docs/v357-sequential-profile{suffix}.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
