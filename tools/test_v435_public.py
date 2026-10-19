"""Public-interface isolation checks for final-only v435 (synthetic only)."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.test_v422 import load
from tools.test_v424 import packed_pair
from tools.verify_attention_campaign import equal
from self_check import validate_attention_calibration_result, validate_hif4_params

torch.set_num_threads(4)
parent, child = load(430, 'public'), load(435, 'public')
g = torch.Generator().manual_seed(435)
samples = [{r:packed_pair(n,128 if r=='q' else 64,g) for r in 'qkv'} for n in (4,8,16)]
saved = copy.deepcopy(samples)
rng = torch.get_rng_state().clone()
a = parent.hif4_calibration_attention(samples,2,1,64)
b = child.hif4_calibration_attention(samples,2,1,64)
assert not validate_attention_calibration_result(b, 'v435')
assert equal(a['q_state'], b['q_state']) and equal(a['k_state'], b['k_state'])
assert equal(a['v_state'], b['v_state']['v435_content_parent'])
assert equal(saved, samples) and torch.equal(rng, torch.get_rng_state())
frozen = copy.deepcopy(b)
for role in 'qkv':
    pair = samples[-1][role]
    fn = 'hif4_dynamic_quantize_'+role
    args = (*pair, 2 if role=='q' else 1,64)
    result = getattr(child,fn)(*args,b[role+'_state'])
    assert not validate_hif4_params(result,pair[0].shape,role)
    replay = load(435,'replay_'+role)
    assert equal(result,getattr(replay,fn)(*args,copy.deepcopy(b[role+'_state'])))
    if role != 'v':assert equal(result,getattr(parent,fn)(*args,a[role+'_state']))
    else:
        assert equal(child.hif4_dynamic_quantize_v(*args,a['v_state']),
                     parent.hif4_dynamic_quantize_v(*args,a['v_state']))
assert equal(b,frozen) and equal(saved,samples) and torch.equal(rng,torch.get_rng_state())
for name in ('hif4_calibration_and_quantize_weight','hif4_dynamic_quantize_activation',
             'hif4_dynamic_quantize_q','hif4_dynamic_quantize_k'):
    assert getattr(parent,name).__code__.co_code == getattr(child,name).__code__.co_code
report = {'passed':True,'parent':430,'parent_calibration_preserved':True,
          'qk_and_linear_code_unchanged':True,'unmarked_v_parent_exact':True,
          'inputs_state_rng_unchanged':True,'fresh_role_replay':True,
          'source_sha256':hashlib.sha256((ROOT/'solution/solution_v435.py').read_bytes()).hexdigest()}
(ROOT/'docs/v435-public-mechanisms.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
