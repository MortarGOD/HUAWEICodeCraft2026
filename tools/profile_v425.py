"""Post-push synthetic cost of v425 W search and encoder/state recompilation.

Excludes parent calibration, four public sampling calls and final two dynamic
calls; cannot estimate whole-contest time. Uses actual legal BF16 decoded W/Z.
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
    path = ROOT / 'solution/solution_v425.py'
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    module = load_solution(path)
    generator = torch.Generator().manual_seed(20260907425)
    report = {
        'source_sha256': source_hash, 'threads': 4, 'torch_version': str(torch.__version__),
        'scope': 'Gaussian synthetic data; sparse W search and recompilation separately. Excludes parent pipeline, sampling public dynamic calls, last-calibration dynamic gate, setup and result checks.',
        'online_prediction_allowed': False, 'cases': [],
    }
    for inputs, outputs in ((512, 2048), (4096, 8192)):
        teacher = torch.randn((outputs, inputs), generator=generator).bfloat16().float()
        parent_params = module._quantize_linear_values(teacher, offsets=(0,))
        deployed = module._linear_score_decode(parent_params)
        state = {'linear_shape': 1, 'unrestricted_projection_ridge': 1.e-3}
        module._install_unrestricted_exact_target_state(state, teacher, parent_params)
        state['linear_full_gram'] = deployed.float().T @ deployed.float()
        samples = []
        for _ in range(4):
            raw = torch.randn((32, inputs), generator=generator).bfloat16().float()
            z = module._linear_score_decode(module._quantize_linear_values(raw, offsets=(0,)))
            target = raw @ teacher.T
            samples.append((z, target, target.square().sum().clamp_min(1.e-30)))
        warm = module._v425_fixed_z_refine(parent_params, samples)
        assert warm is not None and warm['weight_params'] is not None, 'Synthetic search failed to exercise accepted-row path'
        assert module._v425_recompile_state(state, warm['weight_params']) is not None
        search_seconds, compile_seconds = [], []
        for _ in range(3):
            started = time.perf_counter()
            result = module._v425_fixed_z_refine(parent_params, samples)
            search_seconds.append(time.perf_counter() - started)
            assert result is not None and result['weight_params'] is not None
            started = time.perf_counter()
            rebuilt = module._v425_recompile_state(state, result['weight_params'])
            compile_seconds.append(time.perf_counter() - started)
            assert rebuilt is not None
        assert bool((result['after'] <= result['before']).all())
        new = result['weight_params']
        assert torch.equal(new['scale_factor'], parent_params['scale_factor'])
        assert torch.equal(new['scale_lv2'], parent_params['scale_lv2'])
        changed = (new['scale_lv3'] != parent_params['scale_lv3']).reshape(outputs, -1).sum(-1)
        assert int(changed.max()) <= 1
        case = {
            'samples': 4, 'rows_per_sample': 32, 'inputs': inputs, 'outputs': outputs,
            'search_seconds': search_seconds, 'search_median_seconds': statistics.median(search_seconds),
            'compile_seconds': compile_seconds, 'compile_median_seconds': statistics.median(compile_seconds),
            'proposed_rows': result['proposed_rows'], 'accepted_rows': result['accepted_rows'],
            'lv3_changes': int(changed.sum()),
            'fixed_z_fold_loss_before': result['before'].sum(-1).tolist(),
            'fixed_z_fold_loss_after': result['after'].sum(-1).tolist(),
            'interpretation': 'Synthetic fixed-Z acceptance only; no final public-dynamic gate or model-bank claim.',
        }
        report['cases'].append(case)
        print(json.dumps(case), flush=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hash
    report.update(complete=True, source_unchanged=True)
    (ROOT / 'docs/v425-overhead.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
