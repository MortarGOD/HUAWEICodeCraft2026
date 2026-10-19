"""Invariants for the offline Attention input-gain/key-centering prototype."""
import ast
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_fitted_dataset import quantize_nvfp4
from probe_attention_temperature import candidate_source, configured
from probe_persistent_covariance import load
from self_check import validate_attention_calibration_result, validate_hif4_params


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(map(lambda x: equal(*x), zip(a, b)))
    return a == b


def main(long=False, production=False):
    torch.set_num_threads(4)
    if long:
        from probe_long_attention_refinement import configured as long_configured, candidate_source as long_source
        m, source = long_configured(4, True), long_source(4, temperature=True)
    else:
        m, source = configured(), candidate_source()
    if production:
        m, source = load(315), (ROOT/'solution/solution_v315.py').read_text()
    parent = load(314)
    generator = torch.Generator().manual_seed(202609315)
    qh, kh, d = 2, 1, 64
    samples = [{r: quantize_nvfp4(torch.randn(t, (qh if r == 'q' else kh)*d,
                         generator=generator)) for r in 'qkv'} for t in (1, 8, 32, 64, 129)]
    states = parent.hif4_calibration_attention(samples, qh, kh, d)
    learned = m.hif4_calibration_attention(samples, qh, kh, d)
    assert not validate_attention_calibration_result(learned, 'learned')
    forced = dict(states, q_state=dict(states['q_state'], temperature_gain=torch.tensor([.8, 1.25]),
                                      gq=True, u=torch.eye(d).unsqueeze(0)),
                  k_state=dict(states['k_state'], key_center_mask=torch.ones(kh, dtype=torch.bool)))
    assert not validate_attention_calibration_result(forced, 'forced')
    checks = 0
    for sample in samples:
        for role, heads in (('q', qh), ('k', kh), ('v', kh)):
            f, original = getattr(m, 'hif4_dynamic_quantize_'+role), getattr(parent, 'hif4_dynamic_quantize_'+role)
            args = (*sample[role], heads, d)
            base = original(*args, states[role+'_state'])
            assert equal(base, f(*args, states[role+'_state'])), ('disabled', role)
            for st in (learned, forced):
                result = f(*args, st[role+'_state'])
                assert not validate_hif4_params(result, sample[role][0].shape, role)
                if sample[role][0].shape[0] == 1 or role == 'v':
                    assert equal(base, result), ('unchanged', role)
                checks += 1
    # Query subsets must retain the full sequence's refinement budget.
    hold = samples[-1]
    ix = torch.arange(0, 129, 8)
    full = m._deq_params(m.hif4_dynamic_quantize_q(*hold['q'], qh, d, forced['q_state']))[ix]
    subset = m._joint_quant_queries(hold, forced['q_state'], qh, kh, d, ix)
    relative_rms = float((full-subset).square().mean().sqrt()/full.square().mean().sqrt())
    assert relative_rms < .02, relative_rms
    # Exact invariance is a raw, pre-quantization claim, not an accuracy guarantee.
    q, k, v = [m.dequantize_nvfp4(*hold[r]).double() for r in 'qkv']
    reference = m._joint_output(q, k, v, qh, kh, d, ix)
    centered = m._joint_output(q, k-k.mean(0), v, qh, kh, d, ix)
    assert torch.allclose(reference, centered, rtol=1e-9, atol=1e-9)
    before = ast.parse((ROOT/'solution/solution_v314.py').read_text())
    after = ast.parse(source)
    changed = {'_dyn_qk', '_joint_quant_queries', 'hif4_calibration_attention'}
    for old, new in zip(before.body, after.body):
        if isinstance(old, ast.Expr) and isinstance(old.value, ast.Constant) and isinstance(old.value.value, str):
            continue  # Version header, not executable code.
        if isinstance(old, (ast.FunctionDef, ast.ClassDef)) and old.name in changed:
            continue
        if long and isinstance(old, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='ATTN_REFINE_MAX_T' for t in old.targets):
            continue
        assert ast.dump(old) == ast.dump(new), getattr(old, 'name', type(old).__name__)
    banned = {'os','pathlib','io','tempfile','shutil','subprocess','socket','ctypes','pickle','marshal',
              'importlib','zipimport','pkgutil','runpy','mmap','resource','pty','tty','sys','builtins'}
    for n in ast.walk(after):
        if isinstance(n, ast.Import):
            assert all(a.name.split('.')[0] not in banned for a in n.names)
        if isinstance(n, ast.ImportFrom):
            assert (n.module or '').split('.')[0] not in banned
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            assert n.func.id not in {'exec', 'eval', 'open', '__import__'}
    report = dict(passed=True, dynamic_format_checks=checks,
                  query_subset_relative_rms=relative_rms,
                  linear_and_unmodified_functions_ast_equal=True,
                  note=('Actual standalone submission; checks do not qualify an online score prediction.'
                        if production else 'Offline prototype only; these checks do not qualify an online score prediction.'))
    output = 'docs/v315-long-temperature-invariants.json' if long else 'docs/v315-temperature-invariants.json'
    if production:
        output = 'docs/v315-production-invariants.json'
    (ROOT/output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main(long='--long' in sys.argv or '--production' in sys.argv,
         production='--production' in sys.argv)
