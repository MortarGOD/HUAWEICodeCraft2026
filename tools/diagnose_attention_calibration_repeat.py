"""Distinguish source effects from same-source calibration instability."""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.check_role_decoy_calibration import role_samples
from tools.verify_attention_campaign import equal


def differences(a,b,path=''):
    if equal(a,b):return []
    if isinstance(a,dict) and isinstance(b,dict):
        out=[]
        for key in sorted(set(a)|set(b)):
            out.extend(differences(a.get(key),b.get(key),path+'/'+key))
        return out
    if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and a.shape==b.shape:
        return [{'path':path,'shape':list(a.shape),'changed':int((a!=b).sum()),'max_abs':float((a.float()-b.float()).abs().max())}]
    return [{'path':path,'a':str(a),'b':str(b)}]


def main():
    torch.set_num_threads(4)
    group=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)[49]
    samples=role_samples(group['calib'],'q','block_resample',829+ord('q'))
    initial=torch.get_rng_state().clone();states=[];calls=[]
    for version in (422,429,422,429):
        m=load(version,'repeat');count=[0]
        if version==429:
            old=m._v426_quotient_refine
            def wrapped(*a,**kw):count[0]+=1;return old(*a,**kw)
            m._v426_quotient_refine=wrapped
        torch.set_rng_state(initial)
        result=m.hif4_calibration_attention(copy.deepcopy(samples),group['q_num_heads'],group['kv_num_heads'],group['head_dim'])
        assert torch.equal(initial,torch.get_rng_state())
        states.append(result['q_state']);calls.append(count[0])
        print('calibrated',version,'quotient calls',count[0],flush=True)
    pairs=[(0,1),(0,2),(1,3),(2,3)]
    result={'group':49,'role':'q','mode':'block_resample','versions':[422,429,422,429],
            'quotient_calls_during_calibration':calls,'comparisons':[{'indices':[i,j],'equal':equal(states[i],states[j]),'differences':differences(states[i],states[j])} for i,j in pairs]}
    (ROOT/'docs/attention-calibration-repeat-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    with torch.inference_mode():main()
