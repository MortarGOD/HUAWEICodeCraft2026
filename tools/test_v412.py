"""Final-state V ablations: exact boundaries, active nested states and official checks."""
import ast
import copy
import tempfile
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import load_solution
from self_check import self_check, validate_attention_calibration_result, validate_hif4_params
from tools.test_dual_operator_isolation import equal


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    group = torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt', weights_only=False)[10]
    qh, kh, d = (group[k] for k in ('q_num_heads', 'kv_num_heads', 'head_dim'))
    base_source = (ROOT/'solution/solution_v400.py').read_text()
    for version, drop in ((412, 'vr_vm'),):
        path = ROOT/f'solution/solution_v{version}.py'
        source = path.read_text()
        assert source.startswith(base_source)
        for n in ast.walk(ast.parse(source)):
            if isinstance(n, ast.Import):
                assert all(a.name.split('.')[0] in {'torch', 'numpy', 'typing', 'threading'} for a in n.names)
            if isinstance(n, ast.ImportFrom):
                assert n.module.split('.')[0] in {'__future__', 'torch', 'numpy', 'typing', 'threading'}
        m = load_solution(path)
        parent = m._va_parent_calibration
        capture = []
        def observed(*args):
            result = parent(*args)
            capture.append(copy.deepcopy(result))
            return result
        m._va_parent_calibration = observed
        states = m.hif4_calibration_attention(group['calib'], qh, kh, d)
        assert equal(states['q_state'], capture[0]['q_state'])
        assert equal(states['k_state'], capture[0]['k_state'])
        assert not validate_attention_calibration_result(states, 'ablated')
        before = copy.deepcopy(states)
        rng = torch.random.get_rng_state().clone()
        for sample in group['test']:
            fresh = load_solution(path)
            out = fresh.hif4_dynamic_quantize_v(*sample['v'], kh, d, states['v_state'])
            base = load_solution(ROOT/'solution/solution_v400.py')
            expected = base.hif4_dynamic_quantize_v(*sample['v'], kh, d, states['v_state'])
            assert equal(out, expected)
            assert not validate_hif4_params(out, sample['v'][0].shape, 'v')
        assert equal(states, before) and torch.equal(rng, torch.random.get_rng_state())

        # Force all three real dynamic layers active to cover branches that calibration may reject.
        q, s = group['test'][-1]['v']
        q, s = q[:32, :64].contiguous(), s[:32, :4].contiguous()
        x = m.dequantize_nvfp4(q, s).float()
        kernel = dict(center=torch.zeros(64), projection=torch.eye(64)[:, :2].contiguous(),
                      kernel=torch.eye(3)[None].contiguous(), active=torch.ones(1, dtype=torch.bool), heads=1)
        vm = dict(vm_active=torch.ones(1, dtype=torch.bool), vm_variance=8., fallback=kernel)
        vr = dict(center=x.mean(0)[None], rms=torch.ones(1, 64),
                  beta=torch.cat((torch.eye(64)*.01, torch.zeros(1,64)))[None].contiguous(),
                  gain=torch.ones(1), fallback=vm)
        nested = dict(vr=vr)
        snapshot = copy.deepcopy(nested)
        pruned = m._va_prune(nested)
        expected_state = copy.deepcopy(nested)
        if drop == 'vr_vm':
            expected_state = expected_state['vr']['fallback']['fallback']
        elif drop == 'vr':
            expected_state = expected_state['vr']['fallback']
        elif drop == 'vm':
            expected_state['vr']['fallback'] = expected_state['vr']['fallback']['fallback']
        else:
            del expected_state['vr']['fallback']['fallback']['kernel']
        assert equal(pruned, expected_state) and equal(nested, snapshot)
        assert equal(m._va_prune(pruned), pruned)
        assert m._va_prune(None) is None
        base = load_solution(ROOT/'solution/solution_v400.py')
        out = m.hif4_dynamic_quantize_v(q, s, 1, 64, pruned)
        assert equal(out, base.hif4_dynamic_quantize_v(q, s, 1, 64, expected_state))
        assert not validate_hif4_params(out, q.shape, 'nested')
        assert equal(nested, snapshot)
        # Official loader sees a single standalone solution.py, not an import wrapper.
        with tempfile.TemporaryDirectory(prefix=f'hif4-v{version}-') as td:
            (Path(td)/'solution.py').symlink_to(path)
            assert self_check(td)
        print('PASS', version, 'full parent prefix, unchanged Q/K, fresh dynamic, active nested pruning, immutable state/RNG, official self_check', flush=True)


if __name__ == '__main__':
    main()

