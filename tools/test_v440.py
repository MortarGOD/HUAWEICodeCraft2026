"""Mechanism checks for short V routing; no dataset performance evaluation."""
import ast
import copy
import json
from pathlib import Path
import sys
import time
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.test_v422 import load
from tools.test_v424 import packed_pair
from tools.verify_attention_campaign import equal
from tools.probe_value_route import evaluate
from self_check import validate_attention_calibration_result, validate_hif4_params
from hif4_quant import dequantize_hif4

torch.set_num_threads(4)
m, base = load(440, 'mechanism'), load(435, 'mechanism')
gen = torch.Generator().manual_seed(440)
record = {'checks': [], 'shapes': []}
source = (ROOT / 'solution/solution_v440.py').read_bytes()
parent = (ROOT / 'solution/solution_v435.py').read_bytes()
assert source.startswith(parent)
assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(ast.parse(source[len(parent):])))
for name in ('hif4_calibration_and_quantize_weight', 'hif4_dynamic_quantize_activation',
             'hif4_dynamic_quantize_q', 'hif4_dynamic_quantize_k'):
    assert getattr(m, name).__code__.co_code == getattr(base, name).__code__.co_code
record['checks'].append('parent_prefix_imports_four_untouched_apis')
for t, kh, d in ((1, 2, 64), (2, 2, 64), (8, 2, 128), (32, 2, 64),
                 (33, 2, 64), (8, 2, 96), (8, 9, 512)):
    q, s = packed_pair(t, kh * d, gen)
    old = None
    for active in (torch.zeros(kh, dtype=torch.bool), torch.ones(kh, dtype=torch.bool),
                   torch.arange(kh) % 2 == 0):
        state = {'v440_parent': old, 'v440_native_heads': active}
        snapshot = copy.deepcopy((q, s, state)); rng = torch.get_rng_state().clone()
        out = m.hif4_dynamic_quantize_v(q, s, kh, d, state)
        assert not validate_hif4_params(out, q.shape, 'v440')
        before = base.hif4_dynamic_quantize_v(q, s, kh, d, old)
        wanted = dequantize_hif4(before, q.shape).reshape(t, kh, d)
        if m._v440_eligible(t, kh, d):
            native = m._dyn_table(m.dequantize_nvfp4(q, s).float(), None, False)
            nv = dequantize_hif4(native, q.shape).reshape(t, kh, d)
            wanted = torch.where(active[None, :, None], nv, wanted)
        assert torch.equal(dequantize_hif4(out, q.shape).reshape(t, kh, d), wanted)
        assert equal(snapshot, (q, s, state)) and torch.equal(rng, torch.get_rng_state())
        fresh = load(440, 'fresh')
        assert equal(out, fresh.hif4_dynamic_quantize_v(q, s, kh, d, copy.deepcopy(state)))
    record['shapes'].append([t, kh, d])
record['checks'].append('mixed_all_none_heads_boundaries_fresh_immutable_rng')
samples = [{r: packed_pair(t, 128 if r == 'q' else 64, gen) for r in 'qkv'} for t in (4, 8, 16)]
snapshot = copy.deepcopy(samples)
captured = []
original = m._v440_parent_calibration
def capture(*args):
    value = original(*args); captured.append(copy.deepcopy(value)); return value
m._v440_parent_calibration = capture
state = m.hif4_calibration_attention(samples, 2, 1, 64)
assert not validate_attention_calibration_result(state, 'v440')
ps = captured[0]
assert equal(state['q_state'], ps['q_state']) and equal(state['k_state'], ps['k_state'])
errors = torch.stack([evaluate(base, s, ps, 2, 1, 64)[0] for s in samples])
expected = (errors[:, 1] / errors[:, 0].clamp_min(1e-30)).mean(0) < .99
observed = state['v_state'].get('v440_native_heads', torch.zeros(1, dtype=torch.bool))
assert torch.equal(expected, observed) and equal(snapshot, samples)
assert m._v440_fit(samples[:1], ps, 2, 1, 64) is ps['v_state']
record['checks'].append('actual_parent_calibration_independent_formula_qk_preserved')
start = time.perf_counter()
m._v440_fit(samples, ps, 2, 1, 64)
record['small_fit_seconds'] = time.perf_counter() - start
record['completed'] = True
(ROOT / 'docs/v440-mechanisms.json').write_text(json.dumps(record, indent=2) + '\n')
print(record)
