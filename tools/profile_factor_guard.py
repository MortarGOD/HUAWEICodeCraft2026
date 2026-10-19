"""Paired state-equivalence and inclusive function timings, offline only."""
import argparse,copy,json,time,statistics
import torch
from probe_v319_value import ROOT,load_solution
from probe_linear_deployed_joint import tree_equal


def profiled(m,samples,parent,dims):
    names=('_am_fit','_am_cal_data','_am_quant','_af_probability','_joint_quant_queries',
           'hif4_dynamic_quantize_k','_am_parent_v','_af_search','_vr_probs')
    saved={n:getattr(m,n) for n in names};stats={n:dict(seconds=0.,calls=0) for n in names}
    for name,func in saved.items():
        def timed(*args,_name=name,_func=func,**kwargs):
            start=time.perf_counter()
            try:return _func(*args,**kwargs)
            finally:
                stats[_name]['seconds']+=time.perf_counter()-start;stats[_name]['calls']+=1
        setattr(m,name,timed)
    start=time.perf_counter()
    try:state=m._am_guard(samples,parent,*dims)
    finally:
        elapsed=time.perf_counter()-start
        for name,func in saved.items():setattr(m,name,func)
    return state,dict(total=elapsed,inclusive=stats)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repeats',type=int,default=1);args=parser.parse_args()
    assert 1<=args.repeats<=5
    torch.set_num_threads(4)
    old=load_solution(ROOT/'solution/solution_v350.py');new=load_solution(ROOT/'solution/solution_v350.py')
    exec(compile((ROOT/'tools/attention_factor_fast_guard.py').read_text(),'fast_guard','exec'),new.__dict__)
    rows=[]
    for dataset,groups in [('local_fitted_models_attention_feedback_v2',(0,1,5,14,19,49)),
                           ('local_fitted_models_covariance_v1',(0,1,24,25,26,49))]:
        bank=torch.load(ROOT/dataset/'attn.pt',weights_only=False)
        for gi in groups:
            g=bank[gi];dims=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
            root=old._am_parent_calibration(g['calib'],*dims);snapshot=copy.deepcopy(root)
            results={};trials=[]
            for repeat in range(args.repeats):
                for label,m in ([('old',old),('new',new)] if (gi+repeat)%2 else [('new',new),('old',old)]):
                    results[label]=profiled(m,g['calib'],root,dims)
                assert tree_equal(results['old'][0],results['new'][0]) and tree_equal(root,snapshot)
                trials.append({k:v[1] for k,v in results.items()})
            for sample in g['test']:
                assert tree_equal(old.hif4_dynamic_quantize_v(*sample['v'],dims[1],dims[2],results['old'][0]['v_state']),
                                  new.hif4_dynamic_quantize_v(*sample['v'],dims[1],dims[2],results['new'][0]['v_state']))
            row=dict(dataset=dataset,group=gi,equal=True,timing={k:v[1] for k,v in results.items()},
                     trials=trials,median_total={k:statistics.median(t[k]['total'] for t in trials) for k in results})
            rows.append(row);print(dataset,gi,{k:v[1]['total'] for k,v in results.items()},flush=True)
    suffix='' if args.repeats==1 else '-repeat'+str(args.repeats)
    (ROOT/f'docs/factor-guard-profile{suffix}.json').write_text(json.dumps(dict(complete=True,repeats=args.repeats,rows=rows,
        warning='Inclusive nested timers cannot be summed. 12 groups/60 tests; paired offline timings, not online prediction.'),indent=2)+'\n')


if __name__=='__main__':main()
