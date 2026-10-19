"""Sequential, same-input algorithm timing, separate from local reference scoring."""
import contextlib
import io
import json
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import evaluate_attention
from probe_persistent_covariance import load


def main():
    bank = torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt', weights_only=False)
    ids = (1, 5, 14, 26, 37, 49)
    groups = [bank[i] for i in ids]
    report = dict(groups=ids, results={},
                  warning='Local CPU; not an online 96-core timing guarantee. Fixed order, single pass.')
    for version in (314, 315):
        m = load(version)
        timing = dict(calibration=0., dynamic=0.)
        depth = 0
        original_calibration = m.hif4_calibration_attention
        def calibration(*args):
            nonlocal depth
            depth += 1
            started = time.perf_counter()
            try:
                return original_calibration(*args)
            finally:
                timing['calibration'] += time.perf_counter()-started
                depth -= 1
        m.hif4_calibration_attention = calibration
        def dynamic(original):
            def wrapped(*args):
                started = time.perf_counter()
                result = original(*args)
                if not depth:
                    timing['dynamic'] += time.perf_counter()-started
                return result
            return wrapped
        for role in 'qkv':
            name = 'hif4_dynamic_quantize_'+role
            setattr(m, name, dynamic(getattr(m, name)))
        with contextlib.redirect_stdout(io.StringIO()):
            scores, total = evaluate_attention(m, groups)
        report['results'][str(version)] = dict(**timing, evaluation=total, score=sum(scores))
        (ROOT/'docs/v315-sequential-profile.json').write_text(json.dumps(report, indent=2)+'\n')
        print(version, timing, 'score', sum(scores), flush=True)


if __name__ == '__main__':
    main()
