"""Grouped-risk matrix proof checks and bounded independent-anchor ablations."""
import argparse,contextlib,copy,io,itertools,json
import torch
from probe_v319_value import ROOT,load_solution
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result,validate_hif4_params


def math_checks(m):
    generator=torch.Generator().manual_seed(348);checks=0;max_error=0.
    for anchors in (0,1,4):
        t=anchors+4
        probability=torch.randn(2,5,t,generator=generator,dtype=torch.float64).softmax(-1)
        gram,kappa=m._pr_moments(probability,anchors)
        odds,bg=m._pr_odds(probability,anchors)
        recovered=m._pr_resolve(dict(odds=odds,background_kappa=bg),t,anchors)
        assert torch.allclose(gram,recovered['gram'],atol=1e-13,rtol=0)
        assert torch.allclose(kappa,recovered['kappa'],atol=1e-13,rtol=0);checks+=2
        permuted=[]
        for order in itertools.permutations(range(anchors,t)):
            pp=probability[:,:,list(range(anchors))+list(order)]
            permuted.append(pp.transpose(1,2)@pp/pp.shape[1])
        h=torch.stack(permuted).mean(0)+torch.eye(t,dtype=torch.float64)*(.001/t**2)
        gradient,diagonal,_=m._pr_gradient(torch.eye(t,dtype=torch.float64)[None].expand(2,-1,-1),gram,kappa,anchors)
        error=float((gradient-h).abs().max());max_error=max(max_error,error)
        assert torch.allclose(gradient,h,atol=1e-13,rtol=0)
        assert torch.allclose(diagonal,h.diagonal(dim1=1,dim2=2),atol=1e-13,rtol=0)
        assert bool((torch.linalg.eigvalsh(h)>=-1e-13).all());checks+=3
        raw=torch.randn(2,t,7,generator=generator,dtype=torch.float64)
        step=torch.rand(2,t,7,generator=generator,dtype=torch.float64)+.1
        base=(raw/step).round().clamp(-7,7)*step
        prior=m._pr_gradient(base-raw,gram,kappa,anchors)[2]
        for rounds in (1,2,8,16):
            out=m._pr_search(raw,base,step,dict(gram=gram,kappa=kappa),anchors,rounds)
            _,_,loss=m._pr_gradient(out-raw,gram,kappa,anchors)
            assert bool((loss<=prior+1e-12).all())
            assert torch.allclose(loss,((out-raw)*(h@(out-raw))).sum((1,2)),atol=1e-12,rtol=0)
            assert torch.allclose(out/step,(out/step).round(),atol=1e-12,rtol=0)
            assert bool(((out/step).abs()<=7+1e-12).all());prior=loss;checks+=4
    for anchors in (1,4):
        t=anchors+4;probability=torch.zeros(2,2,t,dtype=torch.float64)
        probability[0,:,0]=1.;probability[1,:,0]=.3;probability[1,:,-1]=.7
        gram,kappa=m._pr_moments(probability,anchors);odds,bg=m._pr_odds(probability,anchors)
        model=dict(odds=odds,background_kappa=bg)
        recovered=m._pr_resolve(model,t,anchors)
        assert torch.isfinite(odds).all() and torch.isfinite(bg).all()
        assert torch.allclose(gram,recovered['gram'],atol=1e-10,rtol=0);checks+=2
        for length in (anchors+2,4096):
            recovered=m._pr_resolve(model,length,anchors)
            assert torch.isfinite(recovered['gram']).all() and torch.isfinite(recovered['kappa']).all()
            assert bool((torch.linalg.eigvalsh(recovered['gram'])>=-1e-12).all())
            assert bool(((recovered['kappa']>=1)&(recovered['kappa']<=length-anchors)).all());checks+=3
    return dict(checks=checks,max_matrix_error=max_error)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rescale',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    m=load_solution(ROOT/'solution/solution_v324.py')
    exec(compile((ROOT/'tools/attention_position_risk_core.py').read_text(),'position_risk','exec'),m.__dict__)
    m._PR_RESCALE=args.rescale
    report=dict(math=math_checks(m),rows=[]);print(report['math'],flush=True)
    for gi in (0,1,5,14,19,49):
        g=bank[gi];dims=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        parent=m._pr_parent_calibration(g['calib'],*dims)
        m.hif4_calibration_attention=lambda *_,_st=parent:copy.deepcopy(_st)
        with contextlib.redirect_stdout(io.StringIO()):old,_=evaluate_attention(m,[g])
        row=dict(group=gi,baseline=old,modes={})
        for anchors in (0,1,4):
            m._PR_ANCHORS=anchors
            state=m._pr_guard(g['calib'],parent,*dims);debug=copy.deepcopy(m.PR_DEBUG)
            assert not debug['error'] and not validate_attention_calibration_result(state,'position')
            m.hif4_calibration_attention=lambda *_,_st=state:copy.deepcopy(_st)
            with contextlib.redirect_stdout(io.StringIO()):scores,_=evaluate_attention(m,[g])
            for sample in g['test']:
                out=m.hif4_dynamic_quantize_v(*sample['v'],dims[1],dims[2],state['v_state'])
                assert not validate_hif4_params(out,sample['v'][0].shape,'v')
            row['modes'][anchors]=dict(delta=sum(scores)-sum(old),scores=scores,debug=debug)
        report['rows'].append(row);print(gi,{k:(v['delta'],v['debug']['adopted']) for k,v in row['modes'].items()},flush=True)
    report.update(complete=True,totals={k:sum(r['modes'][k]['delta'] for r in report['rows']) for k in (0,1,4)})
    name='position-risk-rescaled-screen.json' if args.rescale else 'position-risk-screen.json'
    (ROOT/'docs'/name).write_text(json.dumps(report,indent=2)+'\n');print(report['totals'],flush=True)


if __name__=='__main__':main()
