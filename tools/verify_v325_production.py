"""Full actual-submission regression against shared v324 calibrations."""
import contextlib,copy,hashlib,io,json
import torch
from build_v325_candidate import ROOT,candidate_source
from probe_v319_value import load_solution
from evaluate_local import evaluate_linear,evaluate_attention
from self_check import validate_linear_calibration_result,validate_attention_calibration_result,validate_hif4_params
from probe_linear_deployed_joint import tree_equal


def main():
    torch.set_num_threads(4);path=ROOT/'solution/solution_v325.py';src=path.read_text()
    assert src==candidate_source()==(ROOT/'solution/solution.py').read_text()
    m=load_solution(path);parent=load_solution(ROOT/'solution/solution_v324.py')
    report=dict(complete=False,sha256=hashlib.sha256(src.encode()).hexdigest(),checks=0,results={},comparisons={},diagnostics=[],score_prediction_allowed=False)
    roots=[];depth=0;original=m._qg_parent_calibration
    def capture(*args):
        result=original(*args);roots.append(copy.deepcopy(result));return result
    m._qg_parent_calibration=capture
    linear_roots=[]
    for name in ('hif4_calibration_and_quantize_weight','hif4_calibration_attention'):
        fn=getattr(m,name)
        def cal(*args,_fn=fn,_name=name):
            nonlocal depth
            depth+=1
            try:out=_fn(*args)
            finally:depth-=1
            errors=validate_linear_calibration_result(out,args[0].shape,'linear') if _name.endswith('weight') else validate_attention_calibration_result(out,'attention')
            assert not errors,errors;report['checks']+=1
            if _name.endswith('weight'):linear_roots.append(copy.deepcopy(out))
            else:
                assert tree_equal(out['k_state'],roots[-1]['k_state']) and tree_equal(out['v_state'],roots[-1]['v_state'])
                report['diagnostics'].append(copy.deepcopy(m.QG_DEBUG))
            return out
        setattr(m,name,cal)
    for role in ('activation','q','k','v'):
        fn=getattr(m,'hif4_dynamic_quantize_'+role)
        def dyn(*args,_fn=fn,_role=role):
            out=_fn(*args)
            if not depth:
                assert not validate_hif4_params(out,args[0].shape,_role);report['checks']+=1
            return out
        setattr(m,'hif4_dynamic_quantize_'+role,dyn)
    for part,fn in (('linear',evaluate_linear),('attn',evaluate_attention)):
        bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2'/(part+'.pt'),weights_only=False);scores=[];baseline=[]
        for gi,g in enumerate(bank):
            with contextlib.redirect_stdout(io.StringIO()):cur,_=fn(m,[g])
            if part=='linear':parent.hif4_calibration_and_quantize_weight=lambda *unused,_root=linear_roots[-1]:copy.deepcopy(_root)
            else:parent.hif4_calibration_attention=lambda *unused,_root=roots[-1]:copy.deepcopy(_root)
            with contextlib.redirect_stdout(io.StringIO()):prev,_=fn(parent,[g])
            scores.extend(cur);baseline.extend(prev)
            if gi%10==9:print(part,gi+1,'delta',sum(scores)-sum(baseline),'checks',report['checks'],flush=True)
        delta=[b-a for a,b in zip(baseline,scores)]
        report['results'][part]=dict(score=sum(scores),cases=scores,parent_score=sum(baseline),parent_cases=baseline)
        report['comparisons'][part]=dict(delta=sum(delta),exact_equal=scores==baseline,up=sum(x>1e-8 for x in delta),down=sum(x< -1e-8 for x in delta),same=sum(abs(x)<=1e-8 for x in delta),worst=min(delta),best=max(delta))
    assert report['checks']==860 and report['comparisons']['linear']['exact_equal']
    assert len(roots)==50 and not any(d.get('error') for d in report['diagnostics'])
    report['complete']=True
    (ROOT/'docs/v325-production-regression.json').write_text(json.dumps(report,indent=2)+'\n');print(report['comparisons'],flush=True)


if __name__=='__main__':main()
