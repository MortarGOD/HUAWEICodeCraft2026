"""Bounded deployable low-rank student screen, not an online score predictor."""
import argparse,contextlib,copy,io,json
import torch
from probe_v319_value import ROOT,load_solution
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result,validate_hif4_params


def math_checks(m):
    generator=torch.Generator().manual_seed(350);checks=0
    for t in (1,4,17,129):
        raw=torch.randn(2,t,9,generator=generator,dtype=torch.float64)
        step=torch.rand(2,t,9,generator=generator,dtype=torch.float64)+.1
        base=(raw/step).round().clamp(-7,7)*step
        p=torch.randn(2,8,t,generator=generator,dtype=torch.float64).softmax(-1)
        for ridge in (.01,.1,1.):
            h=p.transpose(1,2)@p/p.shape[1]+torch.eye(t,dtype=p.dtype)*(ridge/t**2)
            prior=((base-raw)*(h@(base-raw))).sum((1,2))
            for rounds in (1,8,32):
                out=m._af_search(raw,base,step,p,ridge,rounds)
                loss=((out-raw)*(h@(out-raw))).sum((1,2))
                assert bool((loss<=prior+1e-12).all())
                assert torch.allclose(out/step,(out/step).round(),atol=1e-12,rtol=0)
                assert bool(((out/step).abs()<=7+1e-12).all())
                prior=loss;checks+=3
    return checks


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('factor','mean','uniform'),default='factor')
    parser.add_argument('--ridges',default='.01,.1,1.')
    parser.add_argument('--weight-power',type=float,choices=(.5,1.))
    parser.add_argument('--nonlinear',action='store_true')
    parser.add_argument('--coverage',action='store_true')
    parser.add_argument('--coverage-moments',action='store_true')
    parser.add_argument('--dataset',choices=('feedback','covariance'),default='feedback')
    args=parser.parse_args();ridges=list(map(float,args.ridges.split(',')))
    args.coverage=args.coverage or args.coverage_moments
    torch.set_num_threads(4)
    dataset='local_fitted_models_attention_feedback_v2' if args.dataset=='feedback' else 'local_fitted_models_covariance_v1'
    bank=torch.load(ROOT/dataset/'attn.pt',weights_only=False)
    m=load_solution(ROOT/'solution/solution_v324.py')
    for path in ('attention_moment_core.py','attention_factor_risk_core.py'):
        exec(compile((ROOT/'tools'/path).read_text(),path,'exec'),m.__dict__)
    if args.weight_power is not None:
        from weighted_factor_source import weighted_source
        exec(compile(weighted_source(args.weight_power),'weighted_factor','exec'),m.__dict__)
    feature_count=0
    if args.nonlinear:
        from nonlinear_factor_source import nonlinear_source,feature_checks
        exec(compile(nonlinear_source(args.weight_power),'nonlinear_factor','exec'),m.__dict__)
        feature_count=feature_checks(m)
    if args.coverage:
        assert args.weight_power is None and not args.nonlinear and args.mode=='factor'
        from coverage_factor_source import coverage_source,coverage_checks
        exec(compile(coverage_source(),'coverage_factor','exec'),m.__dict__)
        m._CQ_MOMENT_WEIGHTS=args.coverage_moments
        feature_count=coverage_checks(m)
    original=m._af_probability
    if args.mode=='mean':
        m._af_probability=lambda v,st,kh,d:original(v,st,kh,d).mean(1,keepdim=True)
    elif args.mode=='uniform':
        m._af_probability=lambda v,st,kh,d:torch.full((kh,1,v.shape[0]),1./v.shape[0])
    checks=math_checks(m);print('math checks',checks,flush=True)
    rows=[]
    for gi in ((0,1,5,14,19,49) if args.dataset=='feedback' else (0,1,24,25,26,49)):
        group=bank[gi];dims=[group[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        parent=m._am_parent_calibration(group['calib'],*dims)
        m.hif4_calibration_attention=lambda *_,_st=parent:copy.deepcopy(_st)
        with contextlib.redirect_stdout(io.StringIO()):old,_=evaluate_attention(m,[group])
        model=m._am_fit(group['calib'],parent,*dims)
        row=dict(group=gi,baseline=old,modes={})
        for ridge in ridges:
            model['search_ridge']=ridge
            state=m._am_guard(group['calib'],parent,*dims,model=model)
            assert not validate_attention_calibration_result(state,'factor')
            debug=copy.deepcopy(m.AM_DEBUG)
            m.hif4_calibration_attention=lambda *_,_st=state:copy.deepcopy(_st)
            with contextlib.redirect_stdout(io.StringIO()):scores,_=evaluate_attention(m,[group])
            for sample in group['test']:
                out=m.hif4_dynamic_quantize_v(*sample['v'],dims[1],dims[2],state['v_state'])
                assert not validate_hif4_params(out,sample['v'][0].shape,'v')
            row['modes'][ridge]=dict(delta=sum(scores)-sum(old),scores=scores,debug=debug)
        rows.append(row);print(json.dumps(row),flush=True)
    result=dict(complete=True,dataset=dataset,coverage=args.coverage,mode=args.mode,weight_power=args.weight_power,nonlinear=args.nonlinear,feature_checks=feature_count,math_checks=checks,rows=rows,totals={r:sum(row['modes'][r]['delta'] for row in rows) for r in ridges})
    suffix='' if args.mode=='factor' else '-'+args.mode
    if args.weight_power is not None:suffix+='-weighted'+str(args.weight_power)
    if args.nonlinear:suffix+='-nonlinear'
    if args.coverage:suffix+='-coverage'
    if args.coverage_moments:suffix+='-moments'
    if args.dataset!='feedback':suffix+='-'+args.dataset
    (ROOT/f'docs/factor-risk-screen{suffix}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['totals'],flush=True)


if __name__=='__main__':main()
