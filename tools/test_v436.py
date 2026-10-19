"""Mechanism, exact-solve, contract and public-role checks before publication."""
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
from hif4_quant import dequantize_hif4, dequantize_nvfp4, quantize_to_hif4
from self_check import validate_attention_calibration_result, validate_hif4_params

torch.set_num_threads(4)
m = load(436, 'mechanism')
p = load(435, 'mechanism')
g = torch.Generator().manual_seed(436)
record = {'checks': {}, 'reconstructions': [], 'costs': []}

# Independent dense solve verifies the Woodbury algebra in double precision.
x = torch.randn(2, 9, 64, generator=g, dtype=torch.float64)
y = torch.randn(x.shape, generator=g, dtype=torch.float64)
u = torch.randn(x.shape, generator=g, dtype=torch.float64)
prob = m._v436_probability(x)
rho = prob.square().sum((1, 2)) / 9 + 1/81
a = rho + 1/81
chol = torch.linalg.cholesky(prob @ prob.transpose(1, 2) + torch.eye(33,dtype=torch.float64)[None] * a[:, None, None])
z = m._v436_continuous(x, y, u, prob, chol, rho, a)
h = prob.transpose(1, 2) @ prob + torch.eye(9,dtype=torch.float64)[None]/81
expected = torch.linalg.solve(h + torch.eye(9,dtype=torch.float64)[None]*rho[:, None, None], h@x + rho[:, None, None]*(y-u))
assert torch.allclose(z, expected, rtol=1e-11, atol=1e-11)
order = torch.randperm(9, generator=g)
assert torch.allclose(m._v436_probability(x[:, order]), prob[:, :, order], atol=1e-12, rtol=1e-11)
assert torch.allclose(prob.sum(-1).square().sum(-1), torch.ones(2, dtype=torch.float64))
record['checks']['woodbury_dense_and_probability_permutation'] = True

changes = 0
field_changes = {key: 0 for key in ('scale_factor','scale_lv2','scale_lv3','sign','mant')}
for t, kh, d in ((16,1,64),(64,2,128),(257,1,64),(1,1,64)):
    pair = packed_pair(t, kh*d, g)
    raw = dequantize_nvfp4(*pair).float()
    params = m._dyn_table(raw, None, False)
    frozen = copy.deepcopy((raw, params))
    rng = torch.get_rng_state().clone()
    result = m._v436_reconstruct(raw, params, kh, d)
    assert not validate_hif4_params(result, raw.shape, 'v436')
    assert equal(frozen,(raw,params)) and torch.equal(rng,torch.get_rng_state())
    before = dequantize_hif4(params, raw.shape).float()
    after = dequantize_hif4(result, raw.shape).float()
    pr = m._v436_probability(raw.reshape(t,kh,d).transpose(0,1))
    old = m._v436_block_risk(pr,(before-raw).reshape(t,kh,d).transpose(0,1))
    new = m._v436_block_risk(pr,(after-raw).reshape(t,kh,d).transpose(0,1))
    assert bool((new[0] <= old[0]).all()) and bool((new[1] <= old[1]*1.10).all())
    change = int((before != after).sum()); changes += change
    for key in field_changes: field_changes[key] += int((params[key] != result[key]).sum())
    record['reconstructions'].append({'shape':[t,kh,d],'changed_values':change,
        'risk_before':float(old[0].sum()),'risk_after':float(new[0].sum())})
    if t == 1: assert equal(params,result)
assert changes > 0 and field_changes['scale_factor']+field_changes['scale_lv2']+field_changes['scale_lv3'] > 0
record['checks']['legal_full_grid_changes_risk_caps_and_immutability'] = True
record['field_changes'] = field_changes

# Four synthetic shapes: core-only costs, no translation to online time.
for t,kh,d in ((128,2,64),(512,4,128),(2048,4,128),(4096,4,128)):
    raw = torch.randn(t,kh*d,generator=g).bfloat16().float()
    params = m._dyn_table(raw,None,False)
    start = time.perf_counter(); result=m._v436_reconstruct(raw,params,kh,d)
    elapsed = time.perf_counter()-start
    record['costs'].append({'shape':[t,kh,d],'seconds':elapsed,
        'changed_values':int((dequantize_hif4(params,raw.shape)!=dequantize_hif4(result,raw.shape)).sum())})

samples = [{r:packed_pair(n,128 if r=='q' else 64,g) for r in 'qkv'} for n in (4,8,16)]
saved=copy.deepcopy(samples); rng=torch.get_rng_state().clone()
parent=p.hif4_calibration_attention(samples,2,1,64)
child=m.hif4_calibration_attention(samples,2,1,64)
assert not validate_attention_calibration_result(child,'v436')
assert equal(parent['q_state'],child['q_state']) and equal(parent['k_state'],child['k_state'])
assert equal(parent['v_state'],child['v_state']['v436_content_parent'])
frozen=copy.deepcopy(child)
for role in 'qkv':
    args=(*samples[-1][role],2 if role=='q' else 1,64)
    fn='hif4_dynamic_quantize_'+role
    result=getattr(m,fn)(*args,child[role+'_state'])
    assert not validate_hif4_params(result,samples[-1][role][0].shape,role)
    fresh=load(436,'fresh_'+role)
    assert equal(result,getattr(fresh,fn)(*args,copy.deepcopy(child[role+'_state'])))
    if role!='v':assert equal(result,getattr(p,fn)(*args,parent[role+'_state']))
    else:assert equal(getattr(m,fn)(*args,parent['v_state']),getattr(p,fn)(*args,parent['v_state']))
assert equal(saved,samples) and equal(frozen,child) and torch.equal(rng,torch.get_rng_state())
for name in ('hif4_calibration_and_quantize_weight','hif4_dynamic_quantize_activation','hif4_dynamic_quantize_q','hif4_dynamic_quantize_k'):
    assert getattr(m,name).__code__.co_code==getattr(p,name).__code__.co_code
record['checks']['public_parent_preserved_fresh_role_replay_rng'] = True
(ROOT/'docs/v436-mechanisms.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
