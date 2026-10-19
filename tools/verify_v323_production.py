"""Actual-source normal-bank regression against shared v320 parent states."""
import contextlib
import copy
import hashlib
import io
import json
import torch
from probe_v319_value import ROOT,load_solution
from evaluate_local import evaluate_linear,evaluate_attention
from self_check import validate_linear_calibration_result,validate_attention_calibration_result,validate_hif4_params
from test_v320_residual import same


def main():
    torch.set_num_threads(4);file=ROOT/'solution/solution_v323.py'
    assert file.read_bytes()==(ROOT/'solution/solution.py').read_bytes()
    m=load_solution(file);parent=load_solution(ROOT/'solution/solution_v322.py')
    report=dict(complete=False,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),checks=0,results={},comparisons={},diagnostics=[],score_prediction_allowed=False)
    shared=[];states=[];depth=0;original=m._kt_parent_calibration
    def capture(*args):
        result=original(*args);shared.append(copy.deepcopy(result));return result
    m._kt_parent_calibration=capture
    for name in ('hif4_calibration_and_quantize_weight','hif4_calibration_attention'):
        fn=getattr(m,name)
        def cal(*args,_fn=fn,_name=name):
            nonlocal depth
            depth+=1
            try:out=_fn(*args)
            finally:depth-=1
            errors=(validate_linear_calibration_result(out,args[0].shape,'linear') if _name.endswith('weight') else validate_attention_calibration_result(out,'attention'))
            assert not errors,errors;report['checks']+=1
            if _name.endswith('attention'):
                assert '_ts_cache' not in out;states.append(copy.deepcopy(out))
                report['diagnostics'].append(dict(kt=copy.deepcopy(m.KT_DEBUG),ts=copy.deepcopy(m.TS_DEBUG)))
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
    parent_cal=parent.hif4_calibration_attention;parent_states=[]
    def parent_capture(*args):
        state=parent_cal(*args);parent_states.append(copy.deepcopy(state));return state
    parent.hif4_calibration_attention=parent_capture
    for part,fn in (('linear',evaluate_linear),('attn',evaluate_attention)):
        bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2'/(part+'.pt'),weights_only=False);scores=[];baseline=[]
        if part=='linear':
            with contextlib.redirect_stdout(io.StringIO()):scores,_=fn(m,bank)
            baseline=json.loads((ROOT/'docs/v322-production-regression.json').read_text())['results']['linear']['cases']
        else:
            for gi,g in enumerate(bank):
                with contextlib.redirect_stdout(io.StringIO()):current,_=fn(m,[g])
                scores.extend(current);state=shared[-1]
                parent._kt_parent_calibration=lambda *unused,_st=state:copy.deepcopy(_st)
                with contextlib.redirect_stdout(io.StringIO()):previous,_=fn(parent,[g])
                baseline.extend(previous);assert same(states[-1],parent_states[-1])
                if gi%10==9:print(gi+1,'delta',sum(scores)-sum(baseline),'checks',report['checks'],flush=True)
        delta=[b-a for a,b in zip(baseline,scores)]
        report['results'][part]=dict(score=sum(scores),cases=scores,parent_score=sum(baseline),parent_cases=baseline)
        report['comparisons'][part]=dict(delta=sum(delta),exact_equal=scores==baseline,up=sum(x>1e-8 for x in delta),down=sum(x< -1e-8 for x in delta),same=sum(abs(x)<=1e-8 for x in delta))
    assert report['checks']==860 and all(x['exact_equal'] for x in report['comparisons'].values())
    assert len(shared)==50 and not any(d['kt'].get('error') or d['ts'].get('error') for d in report['diagnostics'])
    report['complete']=True
    (ROOT/'docs/v323-production-regression.json').write_text(json.dumps(report,indent=2)+'\n');print(report['comparisons'],flush=True)


if __name__=='__main__':main()
