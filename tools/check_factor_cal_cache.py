"""Whole-calibration cache equivalence and cleanup checks; research harness."""
import argparse,copy,json,time
import torch
from probe_v319_value import ROOT,load_solution
from probe_linear_deployed_joint import tree_equal


def cache_checks(m):
    original=m._ac_parent_key;sizefn=m._ac_bytes;calls=[0];checks=0
    q=torch.ones(2,4);s=torch.ones(2,1);state={'offset':torch.tensor(1.)}
    def fake(q,s,h,d,st):
        calls[0]+=1;return {'value':q+s+st['offset']+h+d}
    def reset():
        m._AC_ACTIVE=dict(targets=[(q,s)],entries=[],bytes=0,peak_bytes=0,hits=0,calls=0)
    m._ac_parent_key=fake;reset()
    try:
        out=m.hif4_dynamic_quantize_k(q,s,1,4,state);expected=out['value'].clone();out['value'].zero_()
        again=m.hif4_dynamic_quantize_k(q,s,1,4,state)
        assert calls[0]==1 and torch.equal(again['value'],expected);checks+=1
        for mutation in (lambda:state['offset'].add_(1),lambda:q.add_(1),lambda:s.add_(1)):
            before=calls[0];mutation();result=m.hif4_dynamic_quantize_k(q,s,1,4,state)
            assert calls[0]==before+1 and torch.equal(result['value'],q+s+state['offset']+5);checks+=1
        before=calls[0];m.hif4_dynamic_quantize_k(q,s,2,4,state)
        assert calls[0]==before+1;checks+=1
        for i in range(20):
            state['offset']=torch.tensor(float(i));m.hif4_dynamic_quantize_k(q,s,1,4,state)
            assert len(m._AC_ACTIVE['entries'])<=6 and m._AC_ACTIVE['bytes']<=67108864
        checks+=1
        reset();m._ac_bytes=lambda _:67108865
        before=calls[0]
        for _ in range(2):m.hif4_dynamic_quantize_k(q,s,1,4,state)
        assert calls[0]==before+2 and not m._AC_ACTIVE['entries'];checks+=1
        m._AC_ACTIVE=None;before=calls[0]
        for _ in range(2):m.hif4_dynamic_quantize_k(q,s,1,4,state)
        assert calls[0]==before+2;checks+=1
    finally:m._AC_ACTIVE=None;m._ac_parent_key=original;m._ac_bytes=sizefn
    return checks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--production',action='store_true');parser.add_argument('--repeat',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);rows=[]
    old=load_solution(ROOT/'solution/solution_v350.py');new=load_solution(ROOT/'solution/solution_v350.py')
    if args.production:new=load_solution(ROOT/'solution/solution_v351.py')
    else:
        for name in ('attention_factor_fast_guard.py','attention_factor_cal_cache.py'):
            exec(compile((ROOT/'tools'/name).read_text(),name,'exec'),new.__dict__)
    checks=cache_checks(new)
    independent=load_solution(ROOT/'solution/solution_v350.py')
    for dataset,groups in [('local_fitted_models_attention_feedback_v2',(0,1,5,14,19,49) if args.production else (0,1,14)),
                           ('local_fitted_models_covariance_v1',(0,1,24,25,26,49) if args.production else (0,24,25))]:
        bank=torch.load(ROOT/dataset/'attn.pt',weights_only=False)
        for gi in groups:
            g=bank[gi];dims=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
            states={};timings={}
            for name,m in ([('old',old),('new',new)] if gi%2 else [('new',new),('old',old)]):
                start=time.perf_counter();states[name]=m.hif4_calibration_attention(g['calib'],*dims)
                timings[name]=time.perf_counter()-start
            assert tree_equal(states['old'],states['new']) and new._AC_ACTIVE is None
            for sample in g['test']:
                for role in ('k','v'):
                    a=getattr(new,'hif4_dynamic_quantize_'+role)(*sample[role],dims[1],dims[2],states['new'][role+'_state'])
                    b=getattr(independent,'hif4_dynamic_quantize_'+role)(*sample[role],dims[1],dims[2],states['old'][role+'_state'])
                    assert tree_equal(a,b)
            row=dict(dataset=dataset,group=gi,equal=True,timing=timings,stats=copy.deepcopy(new._AC_LAST_STATS))
            rows.append(row);print(row,flush=True)
    original=new._ac_parent_calibration
    def fail(*args):raise RuntimeError('intentional cleanup test')
    new._ac_parent_calibration=fail
    try:
        try:new.hif4_calibration_attention(g['calib'],*dims)
        except RuntimeError:pass
        else:raise AssertionError('expected cleanup exception')
        assert new._AC_ACTIVE is None
    finally:new._ac_parent_calibration=original
    name='v351-production-equivalence.json' if args.production else 'factor-cal-cache-check.json'
    if args.repeat:name=name.replace('.json','-repeat.json')
    (ROOT/'docs'/name).write_text(json.dumps(dict(complete=True,cache_checks=checks,rows=rows,exception_cleanup=True,
        warning='See rows for scope; 5 K+V test pairs per group. Whole calibration single paired timings, not a speed guarantee.'),indent=2)+'\n')


if __name__=='__main__':main()
