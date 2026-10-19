"""Cache safety plus actual-file whole-calibration and Q/K/V equivalence."""
import copy
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from self_check import validate_attention_calibration_result,validate_hif4_params


def safety(m):
    q=torch.ones(2,4);s=torch.ones(2,1);state={'x':torch.ones(1)};calls=[0]
    cache=dict(targets=[('k',q,s),('v',q,s)],entries=[],bytes=0,peak_bytes=0,hits={'k':0,'v':0},calls={'k':0,'v':0})
    m._VC_LOCAL.active=cache
    def fake(q,s,h,d,st):calls[0]+=1;return {'value':q+s+st['x']+h+d}
    out=m._vc_call('k',fake,q,s,1,4,state);expected=out['value'].clone();out['value'].zero_()
    assert torch.equal(m._vc_call('k',fake,q,s,1,4,state)['value'],expected) and calls[0]==1
    for mutate in (lambda:state['x'].add_(1),lambda:q.add_(1),lambda:s.add_(1)):
        n=calls[0];mutate();m._vc_call('k',fake,q,s,1,4,state);assert calls[0]==n+1
    for role,h in (('v',1),('k',2)):
        n=calls[0];m._vc_call(role,fake,q,s,h,4,state);assert calls[0]==n+1
    for i in range(12):state['x'].add_(1);m._vc_call('k',fake,q,s,1,4,state)
    assert len(cache['entries'])<=8 and cache['bytes']<=67108864
    sizefn=m._vc_bytes;cache['entries'].clear();cache['bytes']=0
    m._vc_bytes=lambda _:67108865
    try:
        n=calls[0]
        for _ in range(2):m._vc_call('k',fake,q,s,1,4,state)
        assert calls[0]==n+2 and not cache['entries']
    finally:m._vc_bytes=sizefn
    other=[]
    thread=m._Thread(target=lambda:other.append(getattr(m._VC_LOCAL,'active',None)))
    thread.start();thread.join();assert other==[None]
    original=m._vc_parent_calibration
    def fail(*a):raise RuntimeError('cleanup')
    m._vc_parent_calibration=fail
    try:
        try:m.hif4_calibration_attention([],1,1,64)
        except RuntimeError:pass
        else:raise AssertionError('exception missing')
        assert m._VC_LOCAL.active is cache
    finally:m._vc_parent_calibration=original;m._VC_LOCAL.active=None
    n=calls[0];m._vc_call('k',fake,q,s,1,4,state);m._vc_call('k',fake,q,s,1,4,state)
    assert calls[0]==n+2


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    old=load_solution(ROOT/'solution/solution_v375.py');new=load_solution(ROOT/'solution/solution_v376.py')
    safety(new)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    report={'complete':False,'safety':True,'rows':[]}
    for gi in (0,4,14,24,34,47,49):
        g=bank[gi];dims=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        states={};timings={};checks=0
        for name,m in ([('old',old),('new',new)] if gi%2 else [('new',new),('old',old)]):
            start=time.perf_counter();states[name]=m.hif4_calibration_attention(g['calib'],*dims);timings[name]=time.perf_counter()-start
            assert not validate_attention_calibration_result(states[name],name);checks+=1
        equal=new._vc_equal(states['old'],states['new'])
        assert equal,('calibration differs',gi)
        assert getattr(new._VC_LOCAL,'active',None) is None
        hits=copy.deepcopy(new._VC_LAST_STATS)
        for sample in g['test']:
            for role in 'qkv':
                h=dims[0] if role=='q' else dims[1]
                a=getattr(old,'hif4_dynamic_quantize_'+role)(*sample[role],h,dims[2],states['old'][role+'_state'])
                b=getattr(new,'hif4_dynamic_quantize_'+role)(*sample[role],h,dims[2],states['new'][role+'_state'])
                assert new._vc_equal(a,b),(gi,role)
                assert not validate_hif4_params(b,sample[role][0].shape,role);checks+=1
        assert hits==new._VC_LAST_STATS
        row={'group':gi,'state_equal':equal,'checks':checks,'timing':timings,'cache':hits}
        report['rows'].append(row);print(row,flush=True)
        (ROOT/'docs/v376-equivalence.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True
    (ROOT/'docs/v376-equivalence.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
