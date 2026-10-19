"""Actual-file structural and output-equivalence checks for v359."""
import ast
import json
import time

import torch

from build_v359_candidate import ROOT, candidate_source
from probe_v319_value import load_solution
from probe_linear_deployed_joint import tree_equal
from self_check import validate_attention_calibration_result


def definitions(source, names):
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            result[node.name] = ast.dump(node)
    return result


def structural_checks():
    actual = (ROOT / 'solution/solution_v359.py').read_text()
    assert ast.dump(ast.parse(actual)) == ast.dump(ast.parse(candidate_source()))
    old = (ROOT / 'solution/solution_v358.py').read_text()
    frozen = {'hif4_calibration_and_quantize_weight', 'hif4_dynamic_quantize_activation',
              'hif4_dynamic_quantize_q', 'hif4_dynamic_quantize_k'}
    assert definitions(actual, frozen) == definitions(old, frozen)
    banned = set('os pathlib io tempfile shutil subprocess socket ctypes pickle marshal importlib zipimport pkgutil runpy mmap resource pty tty sys builtins'.split())
    for node in ast.walk(ast.parse(actual)):
        if isinstance(node, ast.Import):
            assert all(name.name.split('.')[0] not in banned for name in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module.split('.')[0] not in banned
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ('open', 'exec', 'eval', 'compile', '__import__')
    assert actual.count('_AF_SEARCH_RIDGE=1.') == 2
    assert actual.count('_AF_SEARCH_RIDGE=.1') == 0
    return 6


def main():
    torch.set_num_threads(4)
    checks = structural_checks()
    modules = {version: load_solution(ROOT / f'solution/solution_v{version}.py') for version in (358, 359)}
    rows = []
    for dataset, groups in (
        ('local_fitted_models_attention_feedback_v2', (0, 1, 2, 5, 14, 49)),
        ('local_fitted_models_covariance_v1', (0, 1, 2, 25, 26, 49)),
    ):
        bank = torch.load(ROOT / dataset / 'attn.pt', weights_only=False)
        for group_index in groups:
            group = bank[group_index]
            dims = [group[key] for key in ('q_num_heads', 'kv_num_heads', 'head_dim')]
            states = {}; seconds = {}
            for version, module in modules.items():
                start = time.perf_counter()
                states[version] = module.hif4_calibration_attention(group['calib'], *dims)
                seconds[version] = time.perf_counter() - start
                assert not validate_attention_calibration_result(states[version], f'v{version}')
                checks += 1
            assert tree_equal(states[358], states[359]); checks += 1
            for sample in group['test']:
                left = modules[358].hif4_dynamic_quantize_v(*sample['v'], dims[1], dims[2], states[358]['v_state'])
                right = modules[359].hif4_dynamic_quantize_v(*sample['v'], dims[1], dims[2], states[359]['v_state'])
                assert tree_equal(left, right); checks += 1
            row = {'dataset': dataset, 'group': group_index, 'state_equal': True,
                   'v_outputs_equal': len(group['test']),
                   'calibration_seconds': {str(k): v for k, v in seconds.items()}}
            rows.append(row); print(row, flush=True)
    report = {'complete': True, 'checks': checks, 'rows': rows,
              'v358_seconds': sum(row['calibration_seconds']['358'] for row in rows),
              'v359_seconds': sum(row['calibration_seconds']['359'] for row in rows),
              'warning': 'Paired local 4-thread calibration; not online runtime evidence.'}
    (ROOT / 'docs/v359-production-equivalence.json').write_text(json.dumps(report, indent=2) + '\n')
    print(report, flush=True)


if __name__ == '__main__':
    main()
