"""Post-push fixed-parent-state Q/K pair comparisons; V remains identical."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.verify_attention_campaign import equal,role_calibration
from hif4_quant import dequantize_hif4,dequantize_nvfp4,quantize_to_hif4
from evaluate_local import gqa_attention,mse
from self_check import validate_hif4_params,validate_attention_calibration_result


def mapping(m,st,kh,d,role):
    if st is None:return torch.eye(d)[None].repeat(kh,1,1)
    plain={k:v for k,v in st.items() if k!='temperature_gain'}
    return m._kt_input(torch.eye(d).repeat(1,kh),plain,kh,d,role).reshape(d,kh,d).transpose(0,1)


@torch.inference_mode()
def run(name,dataset,indices,isolated=False,decoy='row_permute'):
    torch.set_num_threads(4)
    parent,child=load(435,'438parent'),load(438,'438child')
    groups=torch.load(ROOT/dataset/'attn.pt',weights_only=False)
    path=ROOT/f'docs/v438-attention-{name}.json'
    report={'completed':False,'parent':435,'candidate':438,'dataset':dataset,'groups':indices,
        'role_isolated':isolated,'decoy_mode':decoy,'shared_parent_calibration':True,
        'source_sha256':hashlib.sha256((ROOT/'solution/solution_v438.py').read_bytes()).hexdigest(),
        'cases':[],'calibrations':[],'return_checks':0,'state_checks':0,
        'timing':{'parent_qk':0.,'child_qk':0.,'transport':0.},
        'note':'Same parent state, paired Q/K conversion; no incompatible Q-only/K-only hybrid comparisons. Local timings are not online estimates.'}
    for gi in indices:
        group=groups[gi];snapshot=copy.deepcopy(group)
        qh,kh,d=[group[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        rng=torch.get_rng_state().clone()
        original=role_calibration(parent,group,isolated,decoy)
        start=time.perf_counter()
        converted=dict(original,q_state=child._v438_state(original['q_state'],'q',kh,d),
                       k_state=child._v438_state(original['k_state'],'k',kh,d))
        report['timing']['transport']+=time.perf_counter()-start
        assert not validate_attention_calibration_result(converted,'v438')
        saved=copy.deepcopy((original,converted))
        maps=[[mapping(mod,st[r+'_state'],kh,d,r) for r in 'qk']
              for mod,st in ((parent,original),(child,converted))]
        products=[a@b.transpose(-1,-2) for a,b in maps]
        delta=float((products[0]-products[1]).abs().max())
        assert torch.allclose(products[1],torch.eye(d),atol=5e-5,rtol=5e-5)
        report['calibrations'].append({'group':gi,'parent_product_change':delta,'parent_identity_error':float((products[0]-torch.eye(d)).abs().max()),'child_identity_error':float((products[1]-torch.eye(d)).abs().max()),
            'metric_fallback':{r:converted[r+'_state'].get('canonical_dual_metric_fallback',False)
                               if isinstance(converted[r+'_state'],dict) else False for r in 'qk'}})
        for ti,sample in enumerate(group['test']):
            raw={r:dequantize_nvfp4(*sample[r]).float() for r in 'qkv'}
            base={r:dequantize_hif4(quantize_to_hif4(raw[r]),raw[r].shape) for r in 'qkv'}
            target=gqa_attention(raw['q'],raw['k'],raw['v'],qh,kh,d)
            bm=mse(target,gqa_attention(base['q'],base['k'],base['v'],qh,kh,d))
            vpar=parent.hif4_dynamic_quantize_v(*sample['v'],kh,d,original['v_state'])
            vnew=child.hif4_dynamic_quantize_v(*sample['v'],kh,d,converted['v_state'])
            assert equal(vpar,vnew)
            assert not validate_hif4_params(vpar,raw['v'].shape,'v')
            report['return_checks']+=1
            value=dequantize_hif4(vpar,raw['v'].shape)
            params={};decoded={}
            for arm in (('parent','child') if ti%2==0 else ('child','parent')):
                mod,st=(parent,original) if arm=='parent' else (child,converted)
                params[arm]={};decoded[arm]={}
                for role in 'qk':
                    start=time.perf_counter()
                    ans=getattr(mod,'hif4_dynamic_quantize_'+role)(*sample[role],qh if role=='q' else kh,d,st[role+'_state'])
                    report['timing'][arm+'_qk']+=time.perf_counter()-start
                    assert not validate_hif4_params(ans,raw[role].shape,role)
                    report['return_checks']+=1;params[arm][role]=ans
                    decoded[arm][role]=dequantize_hif4(ans,raw[role].shape)
            errors=[mse(target,gqa_attention(decoded[arm]['q'],decoded[arm]['k'],value,qh,kh,d)) for arm in ('parent','child')]
            report['cases'].append({'group':gi,'test':ti,'shape':[len(raw['v']),qh,kh,d],
                'baseline_mse':bm,'mse':errors,'delta':(errors[0]-errors[1])/bm if bm>0 else 0.,
                'changed_values':{r:int((decoded['parent'][r]!=decoded['child'][r]).sum()) for r in 'qk'}})
            if ti==0:
                for role in 'qk':
                    fresh=load(438,'fresh_'+role)
                    replay=getattr(fresh,'hif4_dynamic_quantize_'+role)(*sample[role],qh if role=='q' else kh,d,copy.deepcopy(converted[role+'_state']))
                    assert equal(replay,params['child'][role]);report['state_checks']+=1
        assert equal(saved,(original,converted)) and equal(snapshot,group)
        assert torch.equal(rng,torch.get_rng_state());report['state_checks']+=3
        path.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'job':name,'group':gi,'delta':sum(r['delta'] for r in report['cases'])}),flush=True)
    rows=report['cases'];report['summary']={'cases':len(rows),'delta':sum(r['delta'] for r in rows),
        'better':sum(r['delta']>1e-12 for r in rows),'worse':sum(r['delta']<-1e-12 for r in rows),
        'equal':sum(abs(r['delta'])<=1e-12 for r in rows),
        'changed_values':{r:sum(v['changed_values'][r] for v in rows) for r in 'qk'}}
    report['completed']=True;path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'job':name,**report['summary']}),flush=True)


if __name__=='__main__':
    run('paired','local_fitted_models_attention_feedback_v2',[0,8,17,25,34,49])
    run('covariance','local_fitted_models_covariance_v1',[0,8,17,25,34,49])
    run('row_permute','local_fitted_models_attention_feedback_v2',[0,24,49],True)
    run('block_resample','local_fitted_models_attention_feedback_v2',[0,24,49],True,'block_resample')
