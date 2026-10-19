"""Archive equivalence and geometry fallback for isolated Attention port."""
import copy
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.test_v424 import packed_pair
from tools.verify_attention_campaign import equal
from evaluate_local import load_solution
from self_check import validate_attention_calibration_result,validate_hif4_params
torch.set_num_threads(4)
meta=json.loads((ROOT/'docs/v439-artifacts.json').read_text())
ref=load_solution(ROOT/meta['reference_path']);m=load(439,'test');parent=load(435,'test')
gen=torch.Generator().manual_seed(439);records=[]
with torch.inference_mode():
    for qh,kh,d,lengths in ((2,1,64,(4,8,16)),(4,2,128,(16,32,64)),(2,2,96,(4,8,16))):
        samples=[{r:packed_pair(n,(qh if r=='q' else kh)*d,gen) for r in 'qkv'} for n in lengths]
        frozen=copy.deepcopy(samples);rng=torch.get_rng_state().clone();start=time.perf_counter()
        actual=m.hif4_calibration_attention(samples,qh,kh,d)
        elapsed=time.perf_counter()-start
        expected=(ref if d%64==0 else parent).hif4_calibration_attention(samples,qh,kh,d)
        assert not validate_attention_calibration_result(actual,'v439')
        for r in 'qkv':
            stripped={k:v for k,v in actual[r+'_state'].items() if k!='_v439_reference'} if d%64==0 else actual[r+'_state']
            assert equal(stripped,expected[r+'_state'])
        saved=copy.deepcopy(actual)
        for r in 'qkv':
            pair=samples[-1][r];nh=qh if r=='q' else kh;fn='hif4_dynamic_quantize_'+r
            ans=getattr(m,fn)(*pair,nh,d,actual[r+'_state'])
            baseline=getattr(ref if d%64==0 else parent,fn)(*pair,nh,d,expected[r+'_state'])
            assert equal(ans,baseline) and not validate_hif4_params(ans,pair[0].shape,r)
            assert equal(ans,getattr(load(439,'fresh_'+r),fn)(*pair,nh,d,copy.deepcopy(actual[r+'_state'])))
        assert equal(saved,actual) and equal(samples,frozen) and torch.equal(rng,torch.get_rng_state())
        records.append({'qh':qh,'kh':kh,'d':d,'calibration_seconds':elapsed,'reference_exact':d%64==0,'parent_fallback':bool(d%64)})
    # Force every reference routing branch, not only naturally selected routes.
    pair=packed_pair(8,128,gen)
    state={'route':2,'radius':3,'kv_num_heads':1,'permutation':torch.arange(127,-1,-1)[None],
           'balance':torch.linspace(.5,2,128)[None],'importance':torch.ones(128),'_v439_reference':True}
    for transform in range(5):
        st=dict(state,transform=transform)
        for role in 'qk':
            fn='hif4_dynamic_quantize_'+role
            assert equal(getattr(m,fn)(*pair,1,128,st),getattr(ref,fn)(*pair,1,128,st))
    for fn in ('hif4_calibration_and_quantize_weight','hif4_dynamic_quantize_activation'):
        assert getattr(m,fn).__code__.co_code==getattr(parent,fn).__code__.co_code
report={'checks':records,'forced_transform_routes':5,'linear_entry_bytecode_equal':True,'state_input_rng_fresh_role_checks':True}
(ROOT/'docs/v439-mechanisms.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
