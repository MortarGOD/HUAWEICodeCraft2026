"""Actual-file full regression with exactly shared v320 calibration states."""
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
from test_v320_residual import same


def main():
    torch.set_num_threads(4);file=ROOT/'solution/solution_v321.py'
    assert file.read_bytes()==(ROOT/'solution/solution.py').read_bytes()
    m=load_solution(file);parent=load_solution(ROOT/'solution/solution_v320.py');control=load_solution(file)
    report=dict(version=321,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),complete=False,checks=0,
                results={},interface_seconds={},diagnostics=[],score_prediction_allowed=False)
    output=ROOT/'docs/v321-production-regression.json'
    if output.exists():
        previous=json.loads(output.read_text())
        if previous.get('complete') and previous.get('sha256')!=report['sha256']:
            archive=ROOT/('docs/v321-production-regression-'+previous['sha256'][:8]+'.json')
            if not archive.exists():archive.write_text(json.dumps(previous,indent=2)+'\n')
    parents=[];finals=[];depth=0
    guard=m._kt_guard
    def timed_guard(*args):
        start=time.perf_counter()
        try:return guard(*args)
        finally:report['new_guard_seconds']=report.get('new_guard_seconds',0)+time.perf_counter()-start
    m._kt_guard=timed_guard
    old=m._kt_parent_calibration
    def capture(*args):
        out=old(*args);parents.append(copy.deepcopy(out));return out
    m._kt_parent_calibration=capture
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
                report['diagnostics'].append(copy.deepcopy(m.KT_DEBUG));finals.append(copy.deepcopy(out))
            return out
        setattr(m,name,cal)
    for role in ('activation','q','k','v'):
        name='hif4_dynamic_quantize_'+role;fn=getattr(m,name)
        def dyn(*args,_fn=fn,_name=name):
            if depth:return _fn(*args)
            start=time.perf_counter();out=_fn(*args)
            report['interface_seconds'][_name]=report['interface_seconds'].get(_name,0)+time.perf_counter()-start
            assert not validate_hif4_params(out,args[0].shape,_name)
            report['checks']+=1;return out
        setattr(m,name,dyn)
    for part,fn in (('linear',evaluate_linear),('attn',evaluate_attention)):
        bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2'/(part+'.pt'),weights_only=False)
        if part=='attn':
            scores=[];base=[];dual_scores=[];budget_checks=[];seconds=0.
            for gi,group in enumerate(bank):
                with contextlib.redirect_stdout(io.StringIO()):current,elapsed=fn(m,[group])
                scores.extend(current);seconds+=elapsed
                state=parents[-1]
                parent.hif4_calibration_attention=lambda *unused,_st=state:copy.deepcopy(_st)
                with contextlib.redirect_stdout(io.StringIO()):previous,_=fn(parent,[group])
                base.extend(previous)
                settings=(m._KT_BUDGET_ROUTE,m._KT_GUARD_ROWS,m._KT_MIN_GAIN)
                depth+=1;m._KT_BUDGET_ROUTE=False;m._KT_GUARD_ROWS=64;m._KT_MIN_GAIN=.005
                try:dual=guard(group['calib'],state,group['q_num_heads'],group['kv_num_heads'],group['head_dim'])
                finally:
                    depth-=1
                    m._KT_BUDGET_ROUTE,m._KT_GUARD_ROWS,m._KT_MIN_GAIN=settings
                equivalent=same(finals[-1],dual)
                budget_checks.append(dict(group=gi,states_equal=equivalent))
                if equivalent:dual_scores.extend(current)
                else:
                    control.hif4_calibration_attention=lambda *unused,_st=dual:copy.deepcopy(_st)
                    with contextlib.redirect_stdout(io.StringIO()):full,_=fn(control,[group])
                    dual_scores.extend(full)
                report['progress']=dict(attention_groups=gi+1,candidate_score=sum(scores),parent_score=sum(base))
                if gi%10==9:
                    (ROOT/'docs/v321-production-regression.json').write_text(json.dumps(report,indent=2)+'\n')
                    print('attention groups',gi+1,'delta',sum(scores)-sum(base),'checks',report['checks'],flush=True)
        else:
            with contextlib.redirect_stdout(io.StringIO()):scores,seconds=fn(m,bank)
        report['results'][part]=dict(score=sum(scores),cases=scores,seconds=seconds)
        print(part,sum(scores),seconds,'checks',report['checks'],flush=True)
        if part=='attn':
            report['paired_parent_attn']=dict(score=sum(base),cases=base)
            report['budget_pruning_comparison']=dict(full_candidates_score=sum(dual_scores),cases=dual_scores,
                delta=sum(scores)-sum(dual_scores),checks=budget_checks)
        else:base=json.loads((ROOT/'docs/v320-production-regression.json').read_text())['results']['linear']['cases']
        delta=[b-a for a,b in zip(base,scores)]
        report.setdefault('comparisons',{})[part]=dict(delta=sum(delta),exact_equal=base==scores,
            up=sum(v>1e-8 for v in delta),down=sum(v< -1e-8 for v in delta),same=sum(abs(v)<=1e-8 for v in delta),worst=min(delta),best=max(delta))
        (ROOT/'docs/v321-production-regression.json').write_text(json.dumps(report,indent=2)+'\n')
        del bank
    assert report['checks']==860 and report['comparisons']['linear']['exact_equal']
    assert len(parents)==50 and all(not d.get('error') for d in report['diagnostics'])
    report['complete']=True
    (ROOT/'docs/v321-production-regression.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['comparisons'],flush=True)


if __name__=='__main__':main()
