"""Score the actual standalone file while validating every public return."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import evaluate_attention, evaluate_linear
from probe_persistent_covariance import load
from self_check import (validate_attention_calibration_result,
                        validate_linear_calibration_result, validate_hif4_params)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--version', type=int, default=315)
    p.add_argument('--dataset', default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--output', default='docs/v315-production-regression.json')
    p.add_argument('--allow-linear-changes', action='store_true',
                   help='Explicit opt-in for a Linear candidate; still validate every return')
    a = p.parse_args()
    m = load(a.version)
    report = dict(version=a.version, dataset=a.dataset, results={}, validated_returns=0,
                  validation_seconds=0., sha256=hashlib.sha256((ROOT/f'solution/solution_v{a.version}.py').read_bytes()).hexdigest(),
                  warning='Modeled activations, not a qualified online score predictor. Evaluation time includes scoring and format checks.')
    depth = 0

    def check(errors):
        assert not errors, errors
        report['validated_returns'] += 1

    def calibration(original, linear):
        def wrapped(*args):
            nonlocal depth
            depth += 1
            try:
                result = original(*args)
            finally:
                depth -= 1
            started = time.perf_counter()
            check(validate_linear_calibration_result(result, args[0].shape, 'weight') if linear
                  else validate_attention_calibration_result(result, 'attention'))
            report['validation_seconds'] += time.perf_counter()-started
            return result
        return wrapped

    def dynamic(original, name):
        def wrapped(*args):
            result = original(*args)
            if depth == 0:
                started = time.perf_counter()
                check(validate_hif4_params(result, args[0].shape, name))
                report['validation_seconds'] += time.perf_counter()-started
            return result
        return wrapped

    m.hif4_calibration_and_quantize_weight = calibration(m.hif4_calibration_and_quantize_weight, True)
    m.hif4_calibration_attention = calibration(m.hif4_calibration_attention, False)
    for role in ('activation', 'q', 'k', 'v'):
        name = 'hif4_dynamic_quantize_'+role
        setattr(m, name, dynamic(getattr(m, name), name))
    for role, evaluator in (('linear', evaluate_linear), ('attn', evaluate_attention)):
        groups = torch.load(ROOT/a.dataset/(role+'.pt'), weights_only=False)
        with contextlib.redirect_stdout(io.StringIO()):
            scores, seconds = evaluator(m, groups)
        report['results'][role] = dict(score=sum(scores), cases=scores, seconds=seconds)
        if role == 'linear' and not a.allow_linear_changes:
            with contextlib.redirect_stdout(io.StringIO()):
                baseline, _ = evaluate_linear(load(314), groups)
            assert scores == baseline, 'Linear changed'
            report['linear_cases_identical_to_v314'] = len(scores)
        print(role, len(scores), sum(scores), seconds, 'checks', report['validated_returns'], flush=True)
        (ROOT/a.output).write_text(json.dumps(report, indent=2)+'\n')
        del groups
    assert report['validated_returns'] == 860, report['validated_returns']
    report['passed'] = True
    (ROOT/a.output).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
