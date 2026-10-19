"""Research harness; file I/O here is never included in a submission."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import evaluate_attention
from probe_persistent_covariance import load


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='local_fitted_models_covariance_v1')
    p.add_argument('--groups', default='0,25')
    p.add_argument('--blends', default='0,0.25,0.75')
    p.add_argument('--output', default='docs/v314-pvgn-probe.json')
    args = p.parse_args()
    groups = torch.load(ROOT / args.dataset / 'attn.pt', weights_only=False)
    if args.groups != 'all':
        groups = [groups[int(i)] for i in args.groups.split(',')]
    results = {}
    for blend in map(float, args.blends.split(',')):
        m = load(310)
        m.GPTQ_BLOCK = 32
        if blend:
            exec(compile((ROOT / 'tools/pv_gn_core.py').read_text(), 'pv_gn_core', 'exec'), m.__dict__)
            m._PVGN_BLEND = blend
        diagnostics = []
        parent = m.hif4_calibration_attention
        def capture(*a):
            st = parent(*a)
            diagnostics.append(dict(getattr(m, 'PVGN_DEBUG', {})))
            return st
        m.hif4_calibration_attention = capture
        torch.manual_seed(20260903)
        with contextlib.redirect_stdout(io.StringIO()):
            scores, seconds = evaluate_attention(m, groups)
        results[str(blend)] = dict(score=sum(scores), seconds=seconds, cases=scores, diagnostics=diagnostics)
        print(blend, sum(scores), seconds, diagnostics, flush=True)
        (ROOT / args.output).write_text(json.dumps(dict(dataset=args.dataset, groups=args.groups,
            warning='Mechanism-only proxy; NOT an online score predictor.', results=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
