"""Actual-source full-bank regression, isolated shared parent calibrations."""
import contextlib,copy,hashlib,io,json
import torch
from probe_v319_value import ROOT,load_solution
from evaluate_local import evaluate_linear,evaluate_attention
from self_check import validate_linear_calibration_result,validate_attention_calibration_result,validate_hif4_params
from probe_linear_deployed_joint import tree_equal

def main():
    torch.set_num_threads(4);file=ROOT/'solution/solution_v324.py';assert file.read_bytes()==(ROOT/'solution/solution.py').read_bytes()
    m=load_solution(file);parent=load_solution(ROOT/'solution/solution_v322.py')
    report=dict(complete=False,dataset='local_fitted_models_attention_feedback_v2',sha256=hashlib.sha256(file.read_bytes()).hexdigest(),checks=0,results={},comparisons={},score_prediction_allowed=False)
    roots=[];attnroots=[];depth=0;original=m._gram_parent_calibration;original_attn=m._kt_parent_calibration
    def capture(*args):
        result=original(*args);root=copy.deepcopy(result);root['activation_state'].pop('linear_full_gram',None);roots.append(root);return result
    def capture_attn(*args):
        result=original_attn(*args);attnroots.append(copy.deepcopy(result));return result
    m._gram_parent_calibration=capture;m._kt_parent_calibration=capture_attn
    for name in ('hif4_calibration_and_quantize_weight','hif4_calibration_attention'):
        fn=getattr(m,name)
        def cal(*args,_fn=fn,_name=name):
            nonlocal depth
            depth+=1
            try:out=_fn(*args)
            finally:depth-=1
            errors=validate_linear_calibration_result(out,args[0].shape,'linear') if _name.endswith('weight') else validate_attention_calibration_result(out,'attention')
            assert not errors,errors;report['checks']+=1
            if _name.endswith('weight'):
                cmp=copy.deepcopy(out);cmp['activation_state'].pop('linear_full_gram',None);assert tree_equal(cmp,roots[-1])
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
            if part=='linear':parent.hif4_calibration_and_quantize_weight=lambda *args,_root=roots[-1]:copy.deepcopy(_root)
            else:parent._kt_parent_calibration=lambda *args,_root=attnroots[-1]:copy.deepcopy(_root)
            with contextlib.redirect_stdout(io.StringIO()):prev,_=fn(parent,[g])
            scores.extend(cur);baseline.extend(prev)
            if gi%10==9:print(part,gi+1,'delta',sum(scores)-sum(baseline),'checks',report['checks'],flush=True)
        delta=[b-a for a,b in zip(baseline,scores)]
        report['results'][part]=dict(score=sum(scores),cases=scores,parent_score=sum(baseline),parent_cases=baseline)
        report['comparisons'][part]=dict(delta=sum(delta),exact_equal=scores==baseline,up=sum(x>1e-8 for x in delta),down=sum(x < -1e-8 for x in delta),same=sum(abs(x)<=1e-8 for x in delta),max_abs_delta=max(map(abs,delta)))
    assert report['checks']==860 and report['comparisons']['attn']['exact_equal']
    assert report['comparisons']['linear']['max_abs_delta']<.001,'Unexpected numerical drift, inspect before release'
    report['complete']=True
    path=ROOT/'docs/v324-production-regression.json'
    if path.exists():
        previous=json.loads(path.read_text())
        if previous.get('sha256')!=report['sha256']:
            archive=path.with_name(path.stem+'-'+previous['sha256'][:8]+'.json')
            if not archive.exists():archive.write_text(json.dumps(previous,indent=2)+'\n')
    path.write_text(json.dumps(report,indent=2)+'\n');print(report['comparisons'],flush=True)
if __name__=='__main__':main()
