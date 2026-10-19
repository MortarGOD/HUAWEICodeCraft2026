"""Post-push synthetic timing of only the extra v423 search stage.

These correlated matrices provide mechanism/cost coverage, not a model bank or
an online runtime prediction. The prior full calibration/search is not timed.
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
from self_check import validate_hif4_params


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    source = ROOT / 'solution/solution_v423.py'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    module = load_solution(source)
    report = {
        'source_sha256': digest, 'threads': 4, 'torch_version': str(torch.__version__),
        'scope': 'Synthetic correlated BF16 matrices; extra helper only, excluding full parent algorithm, preparation and verification.',
        'online_prediction_allowed': False, 'cases': [],
    }
    generator = torch.Generator().manual_seed(20260907423)
    for rows, inputs in ((128, 512), (512, 512), (1024, 1024), (64, 4096)):
        outputs = 2 * inputs
        weight = torch.zeros((outputs, inputs), dtype=torch.bfloat16)
        index = torch.arange(inputs // 2)
        weight[2 * index, index] = 1
        weight[2 * index, index + inputs // 2] = 1
        weight[2 * index + 1, index] = .5
        weight[2 * index + 1, index + inputs // 2] = -.5
        # Exact B.T @ B for the representable two-column blocks above.
        gram = torch.eye(inputs) * 1.25
        gram[index, index + inputs // 2] = .75
        gram[index + inputs // 2, index] = .75
        values = (torch.randn((rows, inputs), generator=generator) * .45).bfloat16()
        codes = (values.float() * 4).round().clamp(-7, 7)
        blocks = inputs // 64
        params = {
            'scale_factor': torch.ones((rows, blocks, 1, 1, 1), dtype=torch.bfloat16),
            'scale_lv2': torch.ones((rows, blocks, 8, 1, 1), dtype=torch.bfloat16),
            'scale_lv3': torch.ones((rows, blocks, 8, 2, 1), dtype=torch.bfloat16),
            'sign': codes.sign().reshape(rows, blocks, 8, 2, 4).bfloat16(),
            'mant': (codes.abs() * .25).reshape(rows, blocks, 8, 2, 4).bfloat16(),
        }
        state = {'v421_prefix_enabled': True, 'v421_teacher_weight': weight,
                 'unrestricted_deployed_weight': weight, 'linear_full_gram': gram}
        module._v423_refine_linear_pairs(params, values, state)
        seconds = []
        for _ in range(3):
            started = time.perf_counter()
            result = module._v423_refine_linear_pairs(params, values, state)
            seconds.append(time.perf_counter() - started)
        assert not validate_hif4_params(result, values.shape, 'profile-v423')
        target = values.float() @ weight.float().T
        before = (module._linear_score_decode(params) @ weight.T).float()
        after = (module._linear_score_decode(result) @ weight.T).float()
        old_loss, new_loss = (target - before).square().mean(-1), (target - after).square().mean(-1)
        assert bool((new_loss <= old_loss).all())
        case = {'rows': rows, 'inputs': inputs, 'outputs': outputs,
                'work_product': rows * inputs * outputs,
                'seconds': seconds, 'median_seconds': statistics.median(seconds),
                'improved_rows': int((new_loss < old_loss).sum()),
                'strict_worsened_rows': int((new_loss > old_loss).sum())}
        report['cases'].append(case)
        print(json.dumps(case), flush=True)
    report['source_unchanged'] = hashlib.sha256(source.read_bytes()).hexdigest() == digest
    assert report['source_unchanged']
    report['complete'] = True
    (ROOT / 'docs/v423-search-overhead.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
