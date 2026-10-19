"""Post-push synthetic calibration cost of v424's sampled block learning.

Measures only the added learner, not the full parent calibration/dynamic APIs.
Synthetic data are mechanism and shape coverage, not an online timing estimate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import load_solution


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    path = ROOT / 'solution/solution_v424.py'
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    module = load_solution(path)
    generator = torch.Generator().manual_seed(20260907424)
    report = {'source_sha256': source_hash, 'threads': 4, 'torch_version': str(torch.__version__),
              'scope': 'Synthetic Gaussian A/W; added learner only, excluding parent map, full pipeline, setup and checking.',
              'online_prediction_allowed': False, 'cases': []}
    for rows, inputs, outputs in ((128, 512, 2048), (128, 4096, 8192)):
        activation = torch.randn((rows, inputs), generator=generator).bfloat16().float()
        weight = torch.randn((outputs, inputs), generator=generator).bfloat16().float()
        parent = module._v424_parent_joint_block_transform(activation, weight)
        module._v424_learn_block_transform(activation, weight, parent)
        seconds = []
        for _ in range(3):
            started = time.perf_counter()
            result = module._v424_learn_block_transform(activation, weight, parent)
            seconds.append(time.perf_counter() - started)
        assert result.shape == parent.shape and bool(torch.isfinite(result).all())
        assert torch.equal(result, result.bfloat16().float())
        a_indices = torch.linspace(0, rows - 1, min(128, rows)).long()
        w_indices = torch.linspace(0, outputs - 1, min(256, outputs)).long()
        a, w = activation[a_indices], weight[w_indices]
        target = a @ w.T
        before = module._v424_proxy_pair(a, w, parent, target)
        after = module._v424_proxy_pair(a, w, result, target)
        assert before is not None and after is not None
        assert bool((after[-1] <= before[-1]).all())
        case = {'rows': rows, 'inputs': inputs, 'outputs': outputs,
                'seconds': seconds, 'median_seconds': statistics.median(seconds),
                'changed_blocks': int((parent != result).flatten(1).any(1).sum()),
                'blocks': int(parent.shape[0]), 'parent_losses': before[-1].tolist(),
                'candidate_losses': after[-1].tolist()}
        report['cases'].append(case)
        print(json.dumps(case), flush=True)
    report['source_unchanged'] = hashlib.sha256(path.read_bytes()).hexdigest() == source_hash
    assert report['source_unchanged']
    report['complete'] = True
    (ROOT / 'docs/v424-learning-overhead.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
