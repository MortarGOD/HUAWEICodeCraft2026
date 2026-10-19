"""Full production regression using official validators; only external calls timed."""
import contextlib
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
    torch.set_num_threads(4)
    file=ROOT/'solution/solution_v319.py';m=load_solution(file)
    assert file.read_bytes()==(ROOT/'solution/solution.py').read_bytes()
    report=dict(version=319,dataset='local_fitted_models_attention_feedback_v2',
        sha256=hashlib.sha256(file.read_bytes()).hexdigest(),complete=False,checks=0,results={},interface_seconds={},states=[],
        warning='Checkpoint projections and modeled activations, not captured full-model activations or a qualified online predictor.')
    depth=0
    for name in ('hif4_calibration_and_quantize_weight','hif4_calibration_attention'):
        original=getattr(m,name)
        def calibration(*args,_fn=original,_name=name):
            nonlocal depth
            start=time.perf_counter();depth+=1
            try:out=_fn(*args)
            finally:depth-=1
            elapsed=time.perf_counter()-start
            report['interface_seconds'][_name]=report['interface_seconds'].get(_name,0.)+elapsed
            errors=(validate_linear_calibration_result(out,args[0].shape,'linear') if _name.endswith('weight') else
                    validate_attention_calibration_result(out,'attention'))
            assert not errors,errors
            report['checks']+=1
            if _name.endswith('attention'):
                report['states'].append({role:{key:(bool(st[key]) if not isinstance(st[key],torch.Tensor) else True)
                    for key in ('rot','gq','rf','qs','joint_transform','temperature_gain','kernel') if key in st}
                    for role in ('q_state','k_state','v_state') if isinstance((st:=out[role]),dict)})
            return out
        setattr(m,name,calibration)
    for role in ('activation','q','k','v'):
        name='hif4_dynamic_quantize_'+role;original=getattr(m,name)
        def dynamic(*args,_fn=original,_name=name):
            if depth:return _fn(*args)
            start=time.perf_counter();out=_fn(*args)
            report['interface_seconds'][_name]=report['interface_seconds'].get(_name,0.)+time.perf_counter()-start
            errors=validate_hif4_params(out,args[0].shape,_name);assert not errors,errors
            report['checks']+=1;return out
        setattr(m,name,dynamic)
    for part,fn in (('linear',evaluate_linear),('attn',evaluate_attention)):
        data=torch.load(ROOT/report['dataset']/(part+'.pt'),weights_only=False)
        with contextlib.redirect_stdout(io.StringIO()):scores,seconds=fn(m,data)
        report['results'][part]=dict(score=sum(scores),cases=scores,seconds=seconds)
        (ROOT/'docs/v319-production-regression.json').write_text(json.dumps(report,indent=2)+'\n')
        print(part,sum(scores),seconds,'checks',report['checks'],flush=True)
        del data
    baseline=json.loads((ROOT/'docs/v318-production-regression.json').read_text())
    report['comparisons']={}
    for part in ('linear','attn'):
        a=baseline['results'][part]['cases'];b=report['results'][part]['cases'];diff=[y-x for x,y in zip(a,b)]
        report['comparisons'][part]=dict(delta=sum(diff),exact_equal=a==b,
            up=sum(x>1e-8 for x in diff),down=sum(x< -1e-8 for x in diff),same=sum(abs(x)<=1e-8 for x in diff),
            worst=min(diff),best=max(diff))
    assert report['checks']==860
    assert report['comparisons']['linear']['exact_equal']
    report['complete']=True
    (ROOT/'docs/v319-production-regression.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['comparisons'],report['interface_seconds'],flush=True)


if __name__=='__main__':main()
