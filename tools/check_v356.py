"""Structural, mathematical and paired actual-file checks for v356."""
import ast
import contextlib
import copy
import io
import json
import time

import torch

from attention_hybrid_risk_source import hybrid_risk_source
from build_v356_candidate import ROOT, candidate_source
from probe_v319_value import load_solution
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result


def structural_checks(module):
    actual = (ROOT / 'solution/solution_v356.py').read_text()
    expected = candidate_source()
    assert ast.dump(ast.parse(actual)) == ast.dump(ast.parse(expected))
    banned = set('os pathlib io tempfile shutil subprocess socket ctypes pickle marshal importlib zipimport pkgutil runpy mmap resource pty tty sys builtins'.split())
    tree = ast.parse(actual)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(name.name.split('.')[0] not in banned for name in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module.split('.')[0] not in banned
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ('open', 'exec', 'eval', 'compile', '__import__')
    assert actual.count('_AF_SEARCH_RIDGE=1.') == 2
    assert actual.count('_AF_SEARCH_RIDGE=.1') == 0
    # The concatenated factor implements the intended convex Gram mixture.
    torch.manual_seed(356)
    adaptive = torch.randn(2, 16, 23)
    spectral = torch.randn(2, 4, 23)
    total = 20
    joined = torch.cat((adaptive * (total * .95 / 16) ** .5,
                        spectral * (total * .05 / 4) ** .5), 1)
    got = joined.transpose(1, 2) @ joined / total
    want = .95 * (adaptive.transpose(1, 2) @ adaptive / 16)
    want += .05 * (spectral.transpose(1, 2) @ spectral / 4)
    assert torch.allclose(got, want, atol=2e-6)
    return 5


def main():
    torch.set_num_threads(4)
    modules = {version: load_solution(ROOT / f'solution/solution_v{version}.py') for version in (353, 356)}
    checks = structural_checks(modules[356])
    rows = []
    groups = (0, 1, 2, 3, 4, 5, 6, 14, 19, 24, 25, 26, 49)
    for dataset in ('local_fitted_models_attention_feedback_v2', 'local_fitted_models_covariance_v1'):
        bank = torch.load(ROOT / dataset / 'attn.pt', weights_only=False)
        for group_index in groups:
            group = bank[group_index]
            dims = [group[key] for key in ('q_num_heads', 'kv_num_heads', 'head_dim')]
            versions = {}
            for version, module in modules.items():
                calibration = module.hif4_calibration_attention
                start = time.perf_counter()
                state = calibration(group['calib'], *dims)
                elapsed = time.perf_counter() - start
                assert not validate_attention_calibration_result(state, f'v{version}')
                checks += 1
                if version == 356 and isinstance(state['v_state'], dict) and 'moment' in state['v_state']:
                    model = state['v_state']['moment']
                    assert torch.isfinite(model['spectral_basis']).all()
                    assert model['spectral_basis'].shape[1:] == (4, 64)
                    checks += 2
                module.hif4_calibration_attention = lambda *_, _state=state: copy.deepcopy(_state)
                with contextlib.redirect_stdout(io.StringIO()):
                    scores, _ = evaluate_attention(module, [group])
                module.hif4_calibration_attention = calibration
                versions[str(version)] = {'scores': scores, 'calibration_seconds': elapsed}
            row = {'dataset': dataset, 'group': group_index, 'versions': versions,
                   'delta': sum(versions['356']['scores']) - sum(versions['353']['scores'])}
            rows.append(row)
            print(row, flush=True)
    report = {'complete': True, 'checks': checks, 'rows': rows,
              'delta': sum(row['delta'] for row in rows),
              'warning': 'Modeled activations; paired local regression, not online score/runtime evidence.'}
    (ROOT / 'docs/v356-production-regression.json').write_text(json.dumps(report, indent=2) + '\n')
    print({'checks': checks, 'delta': report['delta']})


if __name__ == '__main__':
    main()
