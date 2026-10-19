"""Estimate exchangeable risk from calibration; no test-dependent tuning."""
import argparse,contextlib,copy,io,json
import torch
from probe_v319_value import ROOT,load_solution
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result,validate_hif4_params


def estimate(m,samples,parent,qh,kh,d):
    rows=[];parameters=[]
    for sample in samples[-3:-1]:
        t=sample['v'][0].shape[0]
        if not 4<=t<=2048 or sample['v'][0].numel()>4194304:continue
        ix=torch.linspace(0,t-1,min(16,t)).long()
        q=m._joint_quant_queries(sample,parent['q_state'],qh,kh,d,ix)
        k=m._deq_params(m.hif4_dynamic_quantize_k(*sample['k'],kh,d,parent['k_state']))
        prob=m._vr_probs(q,k,qh,kh,d)
        s=prob.square().sum(-1).reshape(kh,qh//kh,-1).mean((1,2))
        penalty=((t*s-1)/(1-s).clamp_min(1e-6)).clamp(.01,1e4)
        parameters.append(penalty)
        rows.append(dict(length=t,concentration=s.tolist(),penalty=penalty.tolist()))
    if not parameters:return None,rows
    return torch.stack(parameters).log().mean(0).exp().reshape(kh,1,1).contiguous(),rows


def check_tensor_penalty(m):
    rng=torch.Generator().manual_seed(345);checks=0
    for t in (2,17):
        x=torch.randn(3,t,8,generator=rng,dtype=torch.float64)
        step=torch.rand(3,t,8,generator=rng,dtype=torch.float64)+.1
        base=(x/step).round().clamp(-7,7)*step
        penalty=torch.tensor([.01,1.,100.],dtype=torch.float64).reshape(3,1,1)
        for pairs in (0,2):
            batched=m._ms_search(x,base,step,penalty,pairs)
            split=torch.cat([m._ms_search(x[h:h+1],base[h:h+1],step[h:h+1],float(penalty[h,0,0]),pairs) for h in range(3)])
            assert torch.allclose(batched,split,atol=1e-12,rtol=0)
            def loss(v):
                e=v-x;return e.sum(1).square()+penalty[:,0,:]*e.square().sum(1)
            assert bool((loss(batched)<=loss(base)+1e-10).all())
            assert torch.allclose(batched/step,(batched/step).round(),atol=1e-10,rtol=0)
            checks+=3
    return checks


def main():
    p=argparse.ArgumentParser();p.add_argument('--groups',default='0,1,5,14,19,49')
    p.add_argument('--output',default='docs/concentration-risk-screen.json');a=p.parse_args()
    torch.set_num_threads(4);bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    m=load_solution(ROOT/'solution/solution_v343.py');rows=[]
    checks=check_tensor_penalty(m);print('math',checks,flush=True)
    for gi in map(int,a.groups.split(',')):
        g=bank[gi];dims=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        parent=m._ms_parent_calibration(g['calib'],*dims)
        penalty,calibration=estimate(m,g['calib'],parent,*dims)
        if penalty is None:continue
        m.hif4_calibration_attention=lambda *_,_st=parent:copy.deepcopy(_st)
        with contextlib.redirect_stdout(io.StringIO()):old,_=evaluate_attention(m,[g])
        row=dict(group=gi,calibration=calibration,penalty=penalty.flatten().tolist(),baseline=old,modes={})
        for name,value in (('fixed',.01),('concentration',penalty),('shrink',penalty*.1),('mse',1e6)):
            m._MS_PENALTY=value
            state=m._ms_guard(g['calib'],parent,*dims)
            assert not m.MS_DEBUG['error'] and not validate_attention_calibration_result(state,'concentration')
            m.hif4_calibration_attention=lambda *_,_st=state:copy.deepcopy(_st)
            with contextlib.redirect_stdout(io.StringIO()):scores,_=evaluate_attention(m,[g])
            for sample in g['test']:
                out=m.hif4_dynamic_quantize_v(*sample['v'],dims[1],dims[2],state['v_state'])
                assert not validate_hif4_params(out,sample['v'][0].shape,'v')
            row['modes'][name]=dict(delta=sum(scores)-sum(old),scores=scores,debug=copy.deepcopy(m.MS_DEBUG))
        rows.append(row);print(gi,row['penalty'],{k:(v['delta'],v['debug']['adopted']) for k,v in row['modes'].items()},flush=True)
    report=dict(complete=True,math_checks=checks,rows=rows,totals={k:sum(r['modes'][k]['delta'] for r in rows) for k in ('fixed','concentration','shrink','mse')})
    (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n');print(report['totals'],flush=True)


if __name__=='__main__':main()
