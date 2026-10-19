"""Post-push V-only comparisons with one shared parent calibration per group."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.verify_attention_campaign import equal,role_calibration
from hif4_quant import dequantize_hif4,dequantize_nvfp4,quantize_to_hif4
from evaluate_local import gqa_attention,mse
from self_check import validate_hif4_params,validate_attention_calibration_result


@torch.inference_mode()
def run(name,dataset,indices,isolated=False,decoy='row_permute'):
    torch.set_num_threads(4)
    parent,child=load(435,'v436_parent'),load(436,'v436_child')
    groups=torch.load(ROOT/dataset/'attn.pt',weights_only=False)
    path=ROOT/f'docs/v436-attention-{name}.json'
    record={'completed':False,'parent':435,'candidate':436,'dataset':dataset,
        'groups':indices,'role_isolated':isolated,'decoy_mode':decoy,'cases':[],
        'shared_parent_calibration':True,'public_return_checks':0,'state_checks':0,
        'timing':{'parent_v':0.,'child_v':0.},
        'note':'One parent calibration, child wrapper copies the same state; V-only attribution. Local times are not online estimates.',
        'source_sha256':hashlib.sha256((ROOT/'solution/solution_v436.py').read_bytes()).hexdigest()}
    for gi in indices:
        group=groups[gi];saved=copy.deepcopy(group)
        qh,kh,d=[group[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        rng=torch.get_rng_state().clone()
        state=role_calibration(parent,group,isolated,decoy)
        assert torch.equal(rng,torch.get_rng_state())
        frozen=copy.deepcopy(state)
        child_state={'v436_content_parent':copy.deepcopy(state['v_state'])}
        assert not validate_attention_calibration_result(dict(state,v_state=child_state),'wrapped')
        for ti,sample in enumerate(group['test']):
            raw={r:dequantize_nvfp4(*sample[r]).float() for r in 'qkv'}
            base={r:dequantize_hif4(quantize_to_hif4(raw[r]),raw[r].shape) for r in 'qkv'}
            target=gqa_attention(raw['q'],raw['k'],raw['v'],qh,kh,d)
            bm=mse(target,gqa_attention(base['q'],base['k'],base['v'],qh,kh,d))
            qk={}
            for role in 'qk':
                pars=getattr(parent,'hif4_dynamic_quantize_'+role)(*sample[role],qh if role=='q' else kh,d,state[role+'_state'])
                assert not validate_hif4_params(pars,raw[role].shape,role)
                qk[role]=dequantize_hif4(pars,raw[role].shape)
                record['public_return_checks']+=1
            outputs={};parameters={}
            for arm in (('parent','child') if ti%2==0 else ('child','parent')):
                mod=parent if arm=='parent' else child
                s=state['v_state'] if arm=='parent' else child_state
                start=time.perf_counter()
                pars=mod.hif4_dynamic_quantize_v(*sample['v'],kh,d,s)
                record['timing'][arm+'_v']+=time.perf_counter()-start
                assert not validate_hif4_params(pars,raw['v'].shape,arm)
                record['public_return_checks']+=1
                parameters[arm]=pars;outputs[arm]=dequantize_hif4(pars,raw['v'].shape)
            m0=mse(target,gqa_attention(qk['q'],qk['k'],outputs['parent'],qh,kh,d))
            m1=mse(target,gqa_attention(qk['q'],qk['k'],outputs['child'],qh,kh,d))
            t=len(raw['v']); eligible=2<=t<=4096 and d%64==0 and raw['v'].numel()<=2**20
            row={'group':gi,'test':ti,'shape':[t,kh,d],'eligible':eligible,
                'delta':(m0-m1)/bm if bm>0 else 0.,'baseline_mse':bm,'parent_mse':m0,'child_mse':m1,
                'changed_values':int((outputs['parent']!=outputs['child']).sum()),
                'changed_fields':{k:int((parameters['parent'][k]!=parameters['child'][k]).sum()) for k in parameters['parent']}}
            if eligible:
                prob=child._v436_probability(raw['v'].reshape(t,kh,d).transpose(0,1))
                costs=[]
                for arm in ('parent','child'):
                    error=(outputs[arm].float()-raw['v']).reshape(t,kh,d).transpose(0,1)
                    costs.append(child._v436_block_risk(prob,error))
                assert bool((costs[1][0]<=costs[0][0]).all())
                assert bool((costs[1][1]<=costs[0][1]*1.10).all())
                row['risk']=[float(x[0].sum()) for x in costs]
                row['improved_blocks']=int((costs[1][0]<costs[0][0]).sum())
            else:assert equal(parameters['parent'],parameters['child'])
            if ti==0:
                fresh=load(436,'freshV')
                replay=fresh.hif4_dynamic_quantize_v(*sample['v'],kh,d,copy.deepcopy(child_state))
                assert equal(replay,parameters['child']);record['state_checks']+=1
            record['cases'].append(row)
        assert equal(frozen,state) and equal(child_state['v436_content_parent'],state['v_state'])
        assert equal(saved,group) and torch.equal(rng,torch.get_rng_state())
        record['state_checks']+=3
        path.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({'job':name,'group':gi,'delta':sum(r['delta'] for r in record['cases'])}),flush=True)
    rows=record['cases']
    record['summary']={'cases':len(rows),'delta':sum(r['delta'] for r in rows),
        'better':sum(r['delta']>1e-12 for r in rows),'worse':sum(r['delta']<-1e-12 for r in rows),
        'equal':sum(abs(r['delta'])<=1e-12 for r in rows),
        'eligible':sum(r['eligible'] for r in rows),'changed_cases':sum(r['changed_values']>0 for r in rows),
        'changed_values':sum(r['changed_values'] for r in rows)}
    record['completed']=True;path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'job':name,**record['summary']}),flush=True)


if __name__=='__main__':
    run('paired','local_fitted_models_attention_feedback_v2',[0,8,17,25,34,49])
    run('covariance','local_fitted_models_covariance_v1',[0,8,17,25,34,49])
    run('row_permute','local_fitted_models_attention_feedback_v2',[0,24,49],True)
    run('block_resample','local_fitted_models_attention_feedback_v2',[0,24,49],True,'block_resample')
