"""Fixed-bank test of current-V conditional attention-kernel prediction."""
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
    p.add_argument('--groups', default='0,1,2,3,4,25,26,27,28,29')
    p.add_argument('--ranks', default='0,32,64')
    p.add_argument('--output', default='docs/v314-value-kernel-screen.json')
    args = p.parse_args()
    bank = torch.load(ROOT / 'local_fitted_models_covariance_v1/attn.pt', weights_only=False)
    groups = [bank[int(i)] for i in args.groups.split(',')]
    results = {}
    for rank in map(int, args.ranks.split(',')):
        m = load(310)
        m.GPTQ_BLOCK = 32
        if rank:
            exec(compile((ROOT/'tools/value_kernel_core.py').read_text(), 'value_kernel_core', 'exec'), m.__dict__)
            m._VK_RANK = rank
        diagnostics = []
        parent = m.hif4_calibration_attention
        def capture(*a):
            st = parent(*a)
            diagnostics.append(dict(getattr(m, 'VK_DEBUG', {})))
            return st
        m.hif4_calibration_attention = capture
        with contextlib.redirect_stdout(io.StringIO()):
            scores, sec = evaluate_attention(m, groups)
        results[str(rank)] = dict(score=sum(scores), seconds=sec, cases=scores, diagnostics=diagnostics)
        print(rank, sum(scores), sec, diagnostics, flush=True)
        (ROOT/args.output).write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__': main()
