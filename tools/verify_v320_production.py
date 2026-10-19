"""Verify actual submission, then compare using exactly its parent calibration state."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution,evaluate_linear,evaluate_attention
from self_check import validate_linear_calibration_result,validate_attention_calibration_result,validate_hif4_params


def main():
    torch.set_num_threads(4);file=ROOT/'solution/solution_v320.py'
    assert file.read_bytes()==(ROOT/'solution/solution.py').read_bytes()
    m=load_solution(file);parent=load_solution(ROOT/'solution/solution_v319.py')
    report=dict(version=320,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),complete=False,checks=0,
        results={},interface_seconds={},diagnostics=[],score_prediction_allowed=False,
        warning='Checkpoint projection/modelled-hidden diagnostic bank; not full model forward capture or qualified online predictor.')
    parents=[];depth=0
    for name in ('hif4_calibration_and_quantize_weight','hif4_calibration_attention'):
        fn=getattr(m,name)
        def cal(*args,_fn=fn,_name=name):
            nonlocal depth
            start=time.perf_counter();depth+=1
            try:out=_fn(*args)
            finally:depth-=1
            report['interface_seconds'][_name]=report['interface_seconds'].get(_name,0)+time.perf_counter()-start
            errors=(validate_linear_calibration_result(out,args[0].shape,'linear') if _name.endswith('weight') else
                    validate_attention_calibration_result(out,'attention'))
            assert not errors,errors;report['checks']+=1
            if _name.endswith('attention'):
                assert '_vr_cache' not in out
                st=copy.deepcopy(out)
                if isinstance(st['v_state'],dict) and 'vr' in st['v_state']:
                    st['v_state']=st['v_state']['vr']['fallback']
                parents.append(st);report['diagnostics'].append(copy.deepcopy(m.VR_DEBUG))
            return out
        setattr(m,name,cal)
    for role in ('activation','q','k','v'):
        name='hif4_dynamic_quantize_'+role;fn=getattr(m,name)
        def dyn(*args,_fn=fn,_name=name):
            if depth:return _fn(*args)
            start=time.perf_counter();out=_fn(*args)
            report['interface_seconds'][_name]=report['interface_seconds'].get(_name,0)+time.perf_counter()-start
            errors=validate_hif4_params(out,args[0].shape,_name);assert not errors,errors
            report['checks']+=1;return out
        setattr(m,name,dyn)
    for part,fn in (('linear',evaluate_linear),('attn',evaluate_attention)):
        bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2'/(part+'.pt'),weights_only=False)
        with contextlib.redirect_stdout(io.StringIO()):scores,seconds=fn(m,bank)
        report['results'][part]=dict(score=sum(scores),cases=scores,seconds=seconds)
        print(part,sum(scores),seconds,'checks',report['checks'],flush=True)
        if part=='attn':
            iterator=iter(parents);parent.hif4_calibration_attention=lambda *unused:next(iterator)
            with contextlib.redirect_stdout(io.StringIO()):base,_=evaluate_attention(parent,bank)
            report['paired_parent_attn']=dict(score=sum(base),cases=base)
        else:base=json.loads((ROOT/'docs/v319-production-regression.json').read_text())['results']['linear']['cases']
        diff=[b-a for a,b in zip(base,scores)]
        report.setdefault('comparisons',{})[part]=dict(delta=sum(diff),exact_equal=base==scores,
            up=sum(v>1e-8 for v in diff),down=sum(v< -1e-8 for v in diff),same=sum(abs(v)<=1e-8 for v in diff),worst=min(diff),best=max(diff))
        (ROOT/'docs/v320-production-regression.json').write_text(json.dumps(report,indent=2)+'\n')
        del bank
    assert report['checks']==860 and report['comparisons']['linear']['exact_equal']
    assert all(not d.get('error') for d in report['diagnostics'])
    report['complete']=True
    (ROOT/'docs/v320-production-regression.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['comparisons'],report['interface_seconds'],flush=True)


if __name__=='__main__':main()
