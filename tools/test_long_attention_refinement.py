"""Boundary/format checks for the bounded long-sequence prototype."""
import ast
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_fitted_dataset import quantize_nvfp4
from probe_long_attention_refinement import configured, candidate_source
from probe_persistent_covariance import load
from self_check import validate_hif4_params, validate_attention_calibration_result


def main():
    torch.set_num_threads(4)
    parent, m = load(314), configured(4)
    rng = torch.Generator().manual_seed(31504)
    d, qh, kh = 64, 2, 1
    features = torch.randn(128, d, generator=rng)
    h = (features.T @ features).unsqueeze(0)
    st = dict(kvh=kh, rot=0, gq=1, u=m._upper_cholesky_inv(h), H=h.bfloat16(), rf=1)
    assert not validate_attention_calibration_result(dict(q_state=st, k_state=st, v_state=None), 'forced')
    records = []
    for rows in (1, 128, 1024, 1025, 2048, 4096, 4097):
        sample = {r: quantize_nvfp4(torch.randn(rows, (qh if r == 'q' else kh)*d,
                                              generator=rng)) for r in 'qk'}
        for role, heads in (('q', qh), ('k', kh)):
            f = getattr(m, 'hif4_dynamic_quantize_'+role)
            result = f(*sample[role], heads, d, st)
            assert not validate_hif4_params(result, sample[role][0].shape, role)
            if rows <= 1024 or rows > 4096:
                base = getattr(parent, 'hif4_dynamic_quantize_'+role)(*sample[role], heads, d, st)
                assert all(torch.equal(base[k], result[k]) for k in result), (rows, role)
            if role == 'q' and 1024 < rows <= 4096:
                ix = torch.linspace(0, rows-1, 16).long()
                subset = m._joint_quant_queries(sample, st, qh, kh, d, ix)
                full = m._deq_params(result)[ix]
                relative = float((subset-full).square().mean().sqrt()/full.square().mean().sqrt())
                assert relative < .02, relative
            records.append(dict(rows=rows, role=role, valid=True))
    old = ast.parse((ROOT/'solution/solution_v314.py').read_text())
    new = ast.parse(candidate_source())
    for a, b in zip(old.body, new.body):
        if isinstance(a, ast.FunctionDef) and a.name in ('_dyn_qk', '_joint_quant_queries'):
            continue
        if isinstance(a, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='ATTN_REFINE_MAX_T' for t in a.targets):
            continue
        assert ast.dump(a) == ast.dump(b)
    result = dict(passed=True, checks=records, note='Prototype checks, not new submission self_check.')
    (ROOT/'docs/v315-long-refinement-invariants.json').write_text(json.dumps(result, indent=2)+'\n')
    print('PASS', len(records))


if __name__ == '__main__':
    main()
