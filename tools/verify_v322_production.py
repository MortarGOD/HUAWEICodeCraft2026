"""Actual source, shared parent calibration, all 300 tests and 860 checks."""
import contextlib
import copy
import hashlib
import io
import json
import torch
from probe_v322_translation import ROOT
from probe_v319_value import load_solution
from evaluate_local import evaluate_linear,evaluate_attention
from self_check import validate_linear_calibration_result,validate_attention_calibration_result,validate_hif4_params


def main():
    torch.set_num_threads(4);path=ROOT/'solution/solution_v322.py'
    assert path.read_bytes()==(ROOT/'solution/solution.py').read_bytes()
    m=load_solution(path);parent=load_solution(ROOT/'solution/solution_v321.py')
    report=dict(complete=False,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checks=0,results={},comparisons={},diagnostics=[],score_prediction_allowed=False)
    states=[];depth=0;old=m._ts_parent_calibration
    def capture(*args):
        st=old(*args);states.append(copy.deepcopy({k:v for k,v in st.items() if k!='_ts_cache'}));return st
    m._ts_parent_calibration=capture
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
                assert '_ts_cache' not in out
                report['diagnostics'].append(copy.deepcopy(m.TS_DEBUG))
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
        bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2'/(part+'.pt'),weights_only=False)
        scores=[];baseline=[]
        if part=='linear':
            with contextlib.redirect_stdout(io.StringIO()):scores,seconds=fn(m,bank)
            baseline=json.loads((ROOT/'docs/v321-production-regression.json').read_text())['results']['linear']['cases']
        else:
            for gi,g in enumerate(bank):
                with contextlib.redirect_stdout(io.StringIO()):current,_=fn(m,[g])
                scores.extend(current);st=states[-1]
                parent.hif4_calibration_attention=lambda *unused,_st=st:copy.deepcopy(_st)
                with contextlib.redirect_stdout(io.StringIO()):previous,_=fn(parent,[g])
                baseline.extend(previous)
                if gi%10==9:print(gi+1,'delta',sum(scores)-sum(baseline),'checks',report['checks'],flush=True)
        delta=[b-a for a,b in zip(baseline,scores)]
        report['results'][part]=dict(score=sum(scores),cases=scores,parent_score=sum(baseline),parent_cases=baseline)
        report['comparisons'][part]=dict(delta=sum(delta),exact_equal=scores==baseline,up=sum(x>1e-8 for x in delta),down=sum(x< -1e-8 for x in delta),same=sum(abs(x)<=1e-8 for x in delta),worst=min(delta),best=max(delta))
    assert report['checks']==860 and report['comparisons']['linear']['exact_equal']
    assert len(states)==50 and not any(d.get('error') for d in report['diagnostics'])
    report['complete']=True
    (ROOT/'docs/v322-production-regression.json').write_text(json.dumps(report,indent=2)+'\n');print(report['comparisons'],flush=True)


if __name__=='__main__':main()
