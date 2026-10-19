"""Math, blocked recurrence, legal outputs, and isolation tests."""
import json
import ast
import torch
from probe_v319_gptaq import ROOT, configured, candidate_source
from self_check import validate_hif4_params, validate_frozen_state


def main():
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(319)
    m = configured()
    checks = []
    original = ast.parse((ROOT/'solution/solution_v318.py').read_text())
    candidate = ast.parse(candidate_source())
    for index, node in enumerate(original.body):
        if isinstance(node, ast.FunctionDef) and node.name in ('_dyn_qk', '_joint_quant_queries'):
            continue
        assert ast.dump(node) == ast.dump(candidate.body[index])
    checks.append('all original nodes unchanged except two Q/K dispatch branches')
    for heads, rows, d in ((1, 3, 64), (2, 9, 128), (4, 17, 64)):
        z = torch.randn(heads, d, d, generator=generator, dtype=torch.float64)
        h = z@z.transpose(-1, -2)/d + torch.eye(d)*0.2
        u = torch.linalg.cholesky(torch.linalg.inv(h), upper=True)
        diff = torch.randn(heads, d, d, generator=generator, dtype=torch.float64)*0.003
        p = torch.triu(diff@u.transpose(-1, -2), 1)@u
        x = torch.randn(heads, rows, d, generator=generator, dtype=torch.float64)
        unit = torch.ones_like(x)*1.75
        snapshot = [v.clone() for v in (x, unit, u, p)]
        scalar = m._aq_sweep(x, unit, u, p, block=1)
        for block in (16, 32, d):
            blocked = m._aq_sweep(x, unit, u, p, block=block)
            assert torch.equal(scalar, blocked), (heads, rows, d, block)
            checks.append('blocked recurrence exact codes '+str((heads, rows, d, block)))
        assert all(torch.equal(a, b) for a, b in zip(snapshot, (x, unit, u, p)))
        zero = m._aq_sweep(x.float(), unit.float(), u.float(), torch.zeros_like(p).float())
        assert torch.equal(zero, m._gptq_quantize_batched(x.float(), unit.float(), u.float()))
        # Difference of explicit residual losses equals the centered quadratic
        # with matching ridge. Constant terms need not be carried at runtime.
        raw = torch.randn(heads, 23, d, generator=generator, dtype=torch.float64)
        quant = raw + torch.randn(heads, 23, d, generator=generator, dtype=torch.float64)*0.1
        gram = quant.transpose(-1, -2)@quant
        delta = (raw-quant).transpose(-1, -2)@quant
        ridge = 0.1
        hd = gram+ridge*torch.eye(d)
        center = x@(torch.eye(d)+delta@torch.linalg.inv(hd))
        def explicit(q):
            return ((q@quant.transpose(-1, -2)-x@raw.transpose(-1, -2)).square().sum(-1)
                    +ridge*(q-x).square().sum(-1))
        def quadratic(q):
            error = q-center
            return ((error@hd)*error).sum(-1)
        assert torch.allclose(explicit(scalar)-explicit(x), quadratic(scalar)-quadratic(x), atol=1e-8, rtol=1e-8)
        checks.append('asymmetric objective identity '+str((heads, rows, d)))
    bank = torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt', weights_only=False)
    gi = 1; g = bank[gi]
    qh, kh, d = (g[k] for k in ('q_num_heads', 'kv_num_heads', 'head_dim'))
    parent = m._aq_parent_calibration(g['calib'], qh, kh, d)
    parent_copy = {r: {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in parent[r].items()} for r in ('q_state', 'k_state')}
    st = m._aq_fit(g['calib'], parent, qh, kh, d)
    assert not validate_frozen_state(st, 'gptaq-state')
    sample = g['calib'][-1]; rows = sample['q'][0].shape[0]
    ix = torch.linspace(0, rows-1, min(32, rows)).long()
    for role, nh in (('q', qh), ('k', kh)):
        params = m._dyn_qk(*sample[role], st[role+'_state'], nh, d, role)
        assert not validate_hif4_params(params, sample[role][0].shape, role)
        if role == 'q':
            subset = m._joint_quant_queries(sample, st['q_state'], qh, kh, d, ix)
            assert torch.equal(subset, m._deq_params(params)[ix])
        for key, value in parent_copy[role+'_state'].items():
            actual = parent[role+'_state'][key]
            assert torch.equal(value, actual) if isinstance(value, torch.Tensor) else value == actual
        checks.append(role+' legal and parent unchanged')
        neutral = dict(parent[role+'_state'], aq_p=torch.zeros(kh, d, d),
                       aq_center=torch.eye(d).expand(kh, d, d).contiguous(), aq_mode='triangular')
        base = m._dyn_qk(*sample[role], parent[role+'_state'], nh, d, role)
        zero = m._dyn_qk(*sample[role], neutral, nh, d, role)
        assert base.keys() == zero.keys()
        assert all(torch.equal(base[key], zero[key]) for key in base)
        checks.append(role+' zero compensation params exactly equal')
    # rf=0 parents are allowed to omit H; the guard must not index it blindly.
    without_h = dict(parent)
    for role in 'qk':
        without_h[role+'_state'] = dict(parent[role+'_state'], rf=0)
        without_h[role+'_state'].pop('H', None)
    proposal = m._aq_fit(g['calib'], without_h, qh, kh, d)
    guarded = m._aq_guard(sample, without_h, proposal, qh, kh, d)
    assert not validate_frozen_state(guarded, 'rf0-guard')
    assert all(guarded[role+'_state']['rf'] == 0 for role in 'qk')
    checks.append('missing H with disabled refinement')
    report = dict(passed=True, checks=checks, total=len(checks), actual_group=gi)
    (ROOT/'docs/v319-gptaq-invariants.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
