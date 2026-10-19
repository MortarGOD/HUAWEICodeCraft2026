"""Math checks and shared-parent mean-search ablations (offline only)."""
import argparse,contextlib,copy,io,json,time
import torch
from probe_v319_value import ROOT,load_solution
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result,validate_hif4_params


def math_checks(m):
    rng=torch.Generator().manual_seed(343);checks=0
    for t in (1,2,4,17):
        x=torch.randn(2,t,8,generator=rng,dtype=torch.float64)
        step=torch.rand(2,t,8,generator=rng,dtype=torch.float64)+.1
        base=(x/step).round().clamp(-7,7)*step
        for penalty in (.01,.1,1.):
            def cost(value):
                e=value-x;return e.sum(1).square()+penalty*e.square().sum(1)
            prior=cost(base)
            for pairs in (0,1,2,4):
                value=m._ms_search(x,base,step,penalty,pairs)
                assert bool((cost(value)<=prior+1e-10).all())
                assert torch.allclose(value/step,(value/step).round(),atol=1e-10,rtol=0)
                assert bool(((value/step).abs()<=7+1e-10).all())
                prior=cost(value);checks+=3
    x=torch.tensor([[[.1],[.2]]],dtype=torch.float64);step=torch.tensor([[[1.],[1.25]]],dtype=torch.float64)
    base=torch.zeros_like(x);single=m._ms_search(x,base,step,.01,0);pair=m._ms_search(x,base,step,.01,2)
    def loss(v):return float((v-x).sum().square()+.01*(v-x).square().sum())
    assert torch.equal(single,base) and loss(pair)<loss(single);checks+=1
    return dict(checks=checks,trap_single_loss=loss(single),trap_pair_loss=loss(pair))


def main():
    p=argparse.ArgumentParser();p.add_argument('--groups',default='0,1,5,14,19,49')
    p.add_argument('--output',default='docs/mean-search-screen.json');a=p.parse_args()
    torch.set_num_threads(4);bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    m=load_solution(ROOT/'solution/solution_v324.py')
    exec(compile((ROOT/'tools/attention_mean_search_core.py').read_text(),'mean_search','exec'),m.__dict__)
    report=dict(math=math_checks(m),rows=[]);print(report['math'],flush=True)
    for gi in map(int,a.groups.split(',')):
        g=bank[gi];dims=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        parent=m._ms_parent_calibration(g['calib'],*dims)
        m.hif4_calibration_attention=lambda *_,_st=parent:copy.deepcopy(_st)
        with contextlib.redirect_stdout(io.StringIO()):old,_=evaluate_attention(m,[g])
        row=dict(group=gi,baseline=old,modes={})
        for name,pairs,penalty in (('single',0,.01),('pair2',2,.01),('pair4',4,.01),('regularized',2,.1)):
            m._MS_PAIRS=pairs;m._MS_PENALTY=penalty
            start=time.perf_counter();st=m._ms_guard(g['calib'],parent,*dims);elapsed=time.perf_counter()-start
            assert not m.MS_DEBUG['error'] and not validate_attention_calibration_result(st,'mean')
            m.hif4_calibration_attention=lambda *_,_st=st:copy.deepcopy(_st)
            with contextlib.redirect_stdout(io.StringIO()):scores,_=evaluate_attention(m,[g])
            for s in g['test']:
                out=m.hif4_dynamic_quantize_v(*s['v'],dims[1],dims[2],st['v_state'])
                assert not validate_hif4_params(out,s['v'][0].shape,'v')
            row['modes'][name]=dict(delta=sum(scores)-sum(old),scores=scores,cal_extra=elapsed,debug=copy.deepcopy(m.MS_DEBUG))
        report['rows'].append(row)
        print(gi,{k:(round(v['delta'],6),v['debug']['adopted']) for k,v in row['modes'].items()},flush=True)
    report.update(complete=True,totals={k:sum(r['modes'][k]['delta'] for r in report['rows']) for k in report['rows'][0]['modes']})
    (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n');print(report['totals'],flush=True)


if __name__=='__main__':main()
