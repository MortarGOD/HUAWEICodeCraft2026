"""Long-Key coverage stress, not new independent model samples."""
import argparse
import copy
import hashlib
import json
import time
import torch
from probe_v319_value import ROOT,load_solution
from evaluate_local import score,mse
from hif4_quant import quantize_to_hif4
from self_check import validate_hif4_params


def candidate_source(long_rounds=8):
    source=(ROOT/'solution/solution_v322.py').read_text()
    a='if raw.shape[0]>4096:return _kt_parent_k'
    b="'ts_variant' not in k_state or not 4<=k_quant.shape[0]<=4096:"
    assert source.count(a)==1 and source.count(b)==1
    source=source.replace(a,'if raw.shape[0]>16384 or (raw.shape[0]>4096 and raw.numel()>8388608):return _kt_parent_k')
    source=source.replace(b,b.replace('4096:',"16384 or (k_quant.shape[0]>4096 and k_quant.numel()>8388608):"))
    if long_rounds!=8:
        start=source.index('def _ts_quant(');end=source.index('\ndef _ts_guard(',start)
        body=source[start:end].replace('for _ in range(8):',f'for _ in range(8 if raw.shape[0]<=4096 else {long_rounds}):')
        source=source[:start]+body+source[end:]
    return source


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--groups',default='4,14,19,48,49')
    parser.add_argument('--repeat',type=int,default=4);parser.add_argument('--rounds',type=int,default=8)
    parser.add_argument('--production',action='store_true')
    parser.add_argument('--output',default='docs/v323-long-key-stress.json');args=parser.parse_args()
    torch.set_num_threads(4);bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    parent=load_solution(ROOT/'solution/solution_v322.py');candidate=load_solution(ROOT/'solution/solution_v322.py')
    if args.production:
        from build_v323_candidate import candidate_source as production_source
        source=production_source()
    else:source=candidate_source(args.rounds)
    exec(compile(source,'long-key-candidate','exec'),candidate.__dict__);rows=[]
    for gi in (range(len(bank)) if args.groups=='all' else map(int,args.groups.split(','))):
        group=bank[gi];qh,kh,d=[group[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        state=parent.hif4_calibration_attention(group['calib'],qh,kh,d)
        sample={role:tuple(torch.cat([x]*args.repeat,dim=0) for x in group['test'][-1][role]) for role in 'qkv'}
        t=sample['k'][0].shape[0];ix=torch.linspace(0,t-1,min(t,64)).long()
        rawq=parent.dequantize_nvfp4(sample['q'][0][ix],sample['q'][1][ix]).float()
        rawk=parent.dequantize_nvfp4(*sample['k']).float();rawv=parent.dequantize_nvfp4(*sample['v']).float()
        goal=parent._joint_output(rawq,rawk,rawv,qh,kh,d,ix,query_subset=True)
        base=parent._joint_output(*[parent._deq_params(quantize_to_hif4(x)) for x in (rawq,rawk,rawv)],qh,kh,d,ix,query_subset=True)
        bm=mse(goal,base);q=parent._joint_quant_queries(sample,state['q_state'],qh,kh,d,ix)
        v=parent._deq_params(parent.hif4_dynamic_quantize_v(*sample['v'],kh,d,state['v_state']))
        row=dict(group=gi,rows=t,source_rows=group['test'][-1]['k'][0].shape[0],scores={},key_seconds={})
        for name,m in (('parent',parent),('candidate',candidate)):
            start=time.perf_counter();params=m.hif4_dynamic_quantize_k(*sample['k'],kh,d,copy.deepcopy(state['k_state']))
            row['key_seconds'][name]=time.perf_counter()-start
            assert not validate_hif4_params(params,sample['k'][0].shape,'key')
            out=parent._joint_output(q,parent._deq_params(params),v,qh,kh,d,ix,query_subset=True)
            row['scores'][name]=score(bm,mse(goal,out))
        row['delta']=row['scores']['candidate']-row['scores']['parent'];rows.append(row)
        print(row,flush=True)
    report=dict(complete=True,rows=rows,delta=sum(r['delta'] for r in rows),long_rounds=args.rounds,sha256=hashlib.sha256(source.encode()).hexdigest(),
                score_prediction_allowed=False,warning='Repeated existing projected token sequences, fixed 64 query evaluation, not independent samples or online length-frequency evidence.')
    (ROOT/args.output).write_text(json.dumps(report,indent=2)+'\n');print('delta',report['delta'],flush=True)


if __name__=='__main__':main()
