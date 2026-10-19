"""Actual-file forced-active checks and bounded shared-parent score regression."""
import contextlib,copy,io,json
import torch
from build_output_risk_versions import ROOT,candidate_source
from probe_v319_value import load_solution
from probe_linear_deployed_joint import tree_equal
from build_fitted_dataset import quantize_nvfp4
from evaluate_local import evaluate_attention
from self_check import validate_hif4_params,validate_attention_calibration_result


def main():
    torch.set_num_threads(4);generator=torch.Generator().manual_seed(340)
    report=dict(complete=False,versions={})
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    parent=load_solution(ROOT/'solution/solution_v324.py')
    for version in (339,340):
        assert (ROOT/f'solution/solution_v{version}.py').read_text()==candidate_source(version)
        m=load_solution(ROOT/f'solution/solution_v{version}.py');other=load_solution(ROOT/f'solution/solution_v{version}.py')
        checks=0
        for d in (64,128):
            kh=2
            for t in (1,4,17,129,1025,4097):
                raw=torch.randn(t,kh*d,generator=generator)*.3
                if t==1:raw.zero_()
                packed=quantize_nvfp4(raw)
                if version==339:
                    st=dict(risk_rank=dict(beta=torch.randn(kh,8,generator=generator),scale=torch.ones(kh,8),context=False,
                        active=torch.tensor([True,False]),parent=None))
                else:
                    st=dict(residual_student=dict(beta=torch.randn(kh,d,8,generator=generator)*.05,gain=torch.tensor([1.,0.]),parent=None))
                saved=copy.deepcopy((packed,st));rng=torch.random.get_rng_state().clone()
                out=m.hif4_dynamic_quantize_v(*packed,kh,d,st)
                compare=other.hif4_dynamic_quantize_v(*packed,kh,d,copy.deepcopy(st))
                assert tree_equal(out,compare) and not validate_hif4_params(out,packed[0].shape,'v')
                base=parent.hif4_dynamic_quantize_v(*packed,kh,d,None)
                assert all(torch.equal(x[:,d//64:],base[k][:,d//64:]) for k,x in out.items())
                if t in (1,4097):assert tree_equal(out,base)
                assert tree_equal(saved,(packed,st)) and torch.equal(rng,torch.random.get_rng_state());checks+=5
        rows=[]
        for gi in (0,1,5,14,19,49):
            g=bank[gi];dims=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
            root=parent.hif4_calibration_attention(g['calib'],*dims);saved=copy.deepcopy(root)
            guard=m._rk_guard if version==339 else m._rs_guard
            state=guard(g['calib'],root,*dims);debug=m.RK_DEBUG if version==339 else m.RS_DEBUG
            assert not debug.get('error') and tree_equal(saved,root)
            assert not validate_attention_calibration_result(state,'attention')
            assert tree_equal(root['q_state'],state['q_state']) and tree_equal(root['k_state'],state['k_state'])
            baseline=load_solution(ROOT/'solution/solution_v324.py')
            baseline.hif4_calibration_attention=lambda *_,_st=root:copy.deepcopy(_st)
            m.hif4_calibration_attention=lambda *_,_st=state:copy.deepcopy(_st)
            with contextlib.redirect_stdout(io.StringIO()):old,_=evaluate_attention(baseline,[g]);new,_=evaluate_attention(m,[g])
            rows.append(dict(group=gi,baseline=old,scores=new,delta=sum(new)-sum(old),debug=copy.deepcopy(debug)))
            print(version,gi,rows[-1]['delta'],flush=True);checks+=4
        report['versions'][version]=dict(checks=checks,rows=rows,delta=sum(r['delta'] for r in rows))
    report['complete']=True
    (ROOT/'docs/v339-v340-production-regression.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
