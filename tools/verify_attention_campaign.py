"""Post-push paired public-API evaluation; never estimates an online score."""
import argparse
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
from hif4_quant import dequantize_hif4, dequantize_nvfp4, quantize_to_hif4
from evaluate_local import gqa_attention, mse
from self_check import validate_attention_calibration_result, validate_hif4_params, validate_linear_calibration_result


def equal(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,torch.Tensor):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def save(path,report):
    path.write_text(json.dumps(report,indent=2)+'\n')


def state_differences(a,b,path=''):
    if equal(a,b):return []
    if isinstance(a,dict) and isinstance(b,dict):
        return [row for key in sorted(set(a)|set(b)) for row in state_differences(a.get(key),b.get(key),path+'/'+key)]
    if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and a.shape==b.shape:
        return [{'path':path,'shape':list(a.shape),'changed':int((a!=b).sum()),
                 'max_abs':float((a.float()-b.float()).abs().max())}]
    return [{'path':path,'parent':str(a),'child':str(b)}]


def instrument_quotient(module):
    counts={'calls':0,'seconds':0.,'changed_calls':0,'changed_elements':0,'changed_heads':0,'heads':0,'batch_proposals':0,'nondecreasing_batches':0}
    original=module._v426_quotient_refine
    proposal=module._v426_quotient_proposals
    def prop(*args):
        result=proposal(*args)
        active=result[4];hr=args[3];delta=result[5]
        used=torch.zeros(delta.shape,dtype=torch.long).scatter_add_(0,hr,active.long())>0
        counts['batch_proposals']+=int(used.sum())
        counts['nondecreasing_batches']+=int((used & (~torch.isfinite(delta) | (delta>=0))).sum())
        return result
    def wrapped(*args,**kw):
        start=time.perf_counter();out=original(*args,**kw);counts['seconds']+=time.perf_counter()-start
        counts['calls']+=1;heads=args[4]
        changed=(out!=args[1]).reshape(out.shape[0],heads,-1)
        counts['changed_calls']+=int(bool(changed.any()));counts['changed_elements']+=int(changed.sum())
        counts['changed_heads']+=int(changed.any(0).any(1).sum());counts['heads']+=heads
        return out
    module._v426_quotient_refine=wrapped
    module._v426_quotient_proposals=prop
    return counts


def instrument(module):
    if hasattr(module,'_v426_quotient_refine'):return instrument_quotient(module)
    records={}
    for name in ('_a430_fit','_ac_guard','_a431_tree','_a433_tree','_a433_solve_nodes','_a434_hierarchy','_a434_fine'):
        fn=getattr(module,name,None)
        if fn is None:continue
        rec={'calls':0,'seconds':0.,'exceptions':[],'accepted_heads':[],'temperatures':[],'changed_lv3':0}
        records[name]=rec
        def wrapped(*a,__fn=fn,__name=name,__rec=rec,**kw):
            t=time.perf_counter()
            try:out=__fn(*a,**kw)
            except Exception as exc:
                __rec['exceptions'].append(type(exc).__name__+': '+str(exc));raise
            finally:
                __rec['calls']+=1;__rec['seconds']+=time.perf_counter()-t
            if __name=='_ac_guard':__rec['accepted_heads'].append(out.tolist())
            if __name=='_a430_fit' and out is not None:__rec['temperatures'].append(out.tolist())
            if __name in ('_a431_tree','_a433_tree'):
                __rec['changed_lv3']+=int((out['scale_lv3']!=a[1]['scale_lv3']).sum())
                if __name=='_a433_tree':
                    fields=__rec.setdefault('noncontiguous_input_fields',[])
                    for key,value in a[1].items():
                        if not value.is_contiguous() and key not in fields:fields.append(key)
                    for key in ('scale_lv2','scale_factor'):
                        __rec['changed_'+key]=__rec.get('changed_'+key,0)+int((out[key]!=a[1][key]).sum())
            return out
        setattr(module,name,wrapped)
    return records


def role_calibration(module,group,isolated,decoy_mode="row_permute"):
    qh,kh,d=(group[k] for k in ('q_num_heads','kv_num_heads','head_dim'))
    if not isolated:return module.hif4_calibration_attention(group['calib'],qh,kh,d)
    states={}
    for role in 'qkv':
        from tools.check_role_decoy_calibration import role_samples
        cal=role_samples(group['calib'],role,decoy_mode,829+ord(role))
        result=module.hif4_calibration_attention(cal,qh,kh,d)
        states[role+'_state']=result[role+'_state']
    return states


@torch.inference_mode()
def run(args):
    torch.set_num_threads(4)
    versions=(getattr(args,'parent',422),args.candidate)
    modules=[load(v,'regress') for v in versions]
    counters=instrument(modules[1])
    groups=torch.load(ROOT/args.dataset/'attn.pt',weights_only=False)
    indices=list(range(len(groups))) if args.groups=='all' else [int(x) for x in args.groups.split(',')]
    path=ROOT/args.output
    report={'dataset':args.dataset,'groups':indices,'role_isolated':args.isolated,'source_sha256':hashlib.sha256((ROOT/f'solution/solution_v{args.candidate}.py').read_bytes()).hexdigest(),
        'candidate':args.candidate,'decoy_mode':args.decoy_mode,'completed':False,'cases':[],'calibrations':[],
        'timing':{str(v):{'calibration':0.,'dynamic':0.} for v in versions},'quotient':counters,
        'validated_returns':0,'state_checks':0,'warnings':['Instrumented local timings, not Kunpeng online estimates.','Modeled paired bank or row-permuted dummy peers; online dummy generation is unknown.']}
    save(path,report)
    for number,gi in enumerate(indices):
        g=groups[gi];frozen=copy.deepcopy(g)
        qh,kh,d=(g[k] for k in ('q_num_heads','kv_num_heads','head_dim'))
        initial=torch.get_rng_state().clone();states=[None,None]
        order=[0,1] if number%2==0 else [1,0]
        for arm in order:
            torch.set_rng_state(initial)
            start=time.perf_counter();states[arm]=role_calibration(modules[arm],g,args.isolated,args.decoy_mode)
            report['timing'][str(versions[arm])]['calibration']+=time.perf_counter()-start
            errors=validate_attention_calibration_result(states[arm],'paired');assert not errors,errors
            report['validated_returns']+=1
            assert torch.equal(initial,torch.get_rng_state()),('calibration RNG mutation',gi,arm)
        snapshots=copy.deepcopy(states)
        report['calibrations'].append({'group':gi,'qh':qh,'kh':kh,'dim':d,
            'state_same':{r:equal(states[0][r+'_state'],states[1][r+'_state']) for r in 'qkv'},
            'rf_k':[int(s['k_state'].get('rf',0)) for s in states],
            'new_v_state_keys':sorted(set(states[1]['v_state'] or {})-set(states[0]['v_state'] or {})),
            'unexpected_role_differences':{r:state_differences(states[0][r+'_state'],states[1][r+'_state'])
                                           for r in ('qv' if args.candidate==429 else 'qk')
                                           if not equal(states[0][r+'_state'],states[1][r+'_state'])},
            'kt_mode':[int(s['k_state'].get('kt_mode',0)) for s in states]})
        for ti,sample in enumerate(g['test']):
            raw={r:dequantize_nvfp4(*sample[r]).float() for r in 'qkv'}
            baseline={r:dequantize_hif4(quantize_to_hif4(raw[r]),raw[r].shape) for r in 'qkv'}
            ref=gqa_attention(raw['q'],raw['k'],raw['v'],qh,kh,d)
            bm=mse(ref,gqa_attention(baseline['q'],baseline['k'],baseline['v'],qh,kh,d))
            params=[{},{}];players=[{},{}];costs=[]
            for arm in order:
                module=modules[arm]
                for role in 'vkq':
                    start=time.perf_counter()
                    ans=getattr(module,'hif4_dynamic_quantize_'+role)(*sample[role],qh if role=='q' else kh,d,states[arm][role+'_state'])
                    report['timing'][str(versions[arm])]['dynamic']+=time.perf_counter()-start
                    errors=validate_hif4_params(ans,raw[role].shape,role);assert not errors,errors
                    report['validated_returns']+=1;params[arm][role]=ans
                    players[arm][role]=dequantize_hif4(ans,raw[role].shape)
            for arm in (0,1):costs.append(mse(ref,gqa_attention(players[arm]['q'],players[arm]['k'],players[arm]['v'],qh,kh,d)))
            quality=[1-c/bm for c in costs] if bm>0 else [0.,0.]
            report['cases'].append({'group':gi,'test':ti,'tokens':raw['q'].shape[0],'baseline_mse':bm,'mse':costs,'quality':quality,
                'delta':quality[1]-quality[0],'output_parameters_same':{r:equal(params[0][r],params[1][r]) for r in 'qkv'},
                'dequantized_values_same':{r:torch.equal(players[0][r],players[1][r]) for r in 'qkv'}})
            if ti==0:
                for role in 'qkv':
                    fresh=load(args.candidate,'replay'+role)
                    replay=getattr(fresh,'hif4_dynamic_quantize_'+role)(*sample[role],qh if role=='q' else kh,d,copy.deepcopy(states[1][role+'_state']))
                    assert equal(replay,params[1][role]),('fresh role replay',gi,role)
                    report['state_checks']+=1
        assert equal(snapshots,states),('state mutation',gi)
        assert equal(frozen,g),('input mutation',gi)
        report['state_checks']+=2
        save(path,report)
        print(json.dumps({'group':gi,'done':number+1,'total':len(indices),'delta':sum(x['delta'] for x in report['cases'])}),flush=True)
    vals=report['cases']
    report['summary']={'cases':len(vals),'parent_quality':sum(v['quality'][0] for v in vals),'child_quality':sum(v['quality'][1] for v in vals),
        'delta':sum(v['delta'] for v in vals),'better':sum(v['delta']>1e-12 for v in vals),'worse':sum(v['delta']<-1e-12 for v in vals),
        'equal':sum(abs(v['delta'])<=1e-12 for v in vals),'changed_role_cases':{r:sum(not v['output_parameters_same'][r] for v in vals) for r in 'qkv'}}
    report['completed']=True;save(path,report);print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups',default='all');p.add_argument('--isolated',action='store_true')
    p.add_argument('--output',default='docs/v426-attention-paired.json')
    p.add_argument('--candidate',type=int,required=True)
    p.add_argument('--parent',type=int,default=422)
    p.add_argument('--decoy-mode',choices=['row_permute','block_resample'],default='row_permute')
    run(p.parse_args())
