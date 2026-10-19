"""Finite-budget block-LDL candidate, same parent decoder/calibration."""
import argparse,copy,json,time,ast
import torch
from probe_v319_value import ROOT,load_solution
from hif4_quant import dequantize_hif4,quantize_to_hif4
from self_check import validate_hif4_params

def source():
    s=(ROOT/'solution/solution_v322.py').read_text()
    s=s.replace('    alternate_values: torch.Tensor | None = None,','    alternate_values: torch.Tensor | None = None,\n    ldl_factor: torch.Tensor | None = None,')
    point='''    reconstructed = [
        _dequantize_linear_params(params).reshape(-1, values.shape[-1])
        for params in candidates
    ]'''
    assert s.count(point)==1;s=s.replace(point,'''    if ldl_factor is not None:
        candidates.append(_linear_ldl_proposal(values, ldl_factor))
'''+point)
    point='''            extra_refinement=(shape == -1),
            alternate_values=('''
    assert s.count(point)==1;s=s.replace(point,'''            extra_refinement=(shape == -1),
            ldl_factor=activation_state.get("linear_ldl_factor"),
            alternate_values=(''')
    return s+'\n'+(ROOT/'tools/linear_ldl_core.py').read_text()

def main():
    p=argparse.ArgumentParser();p.add_argument('--groups',default='0,4,6,7,9');p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2');p.add_argument('--output',default='docs/v324-ldl-screen.json');a=p.parse_args()
    torch.set_num_threads(4);m=load_solution(ROOT/'solution/solution_v322.py');new=load_solution(ROOT/'solution/solution_v322.py');exec(compile(source(),'ldl','exec'),new.__dict__)
    bank=torch.load(ROOT/a.dataset/'linear.pt',weights_only=False);rows=[]
    for gi in map(int,a.groups.split(',')):
        g=bank[gi];w=m.dequantize_nvfp4(*g['weight']).float();bw=dequantize_hif4(quantize_to_hif4(w),w.shape).float()
        parent=m.hif4_calibration_and_quantize_weight(*g['weight'],g['calib_activation_list']);b=m._linear_score_decode(parent['weight_params']);state=copy.deepcopy(parent['activation_state'])
        start=time.perf_counter();h=b.float().T@b.float();h.diagonal().add_(.001*h.diagonal().mean());factor=torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)),upper=True);state['linear_ldl_factor']=factor
        compile_seconds=time.perf_counter()-start
        for ti,packed in enumerate(g['test_activation_list']):
            x=m.dequantize_nvfp4(*packed).float();y=x@w.T;bx=dequantize_hif4(quantize_to_hif4(x),x.shape).float();bm=float((y-bx@bw.T).square().mean());r=dict(group=gi,test=ti,scores={},seconds={},compile_seconds=compile_seconds)
            for name,mod,st in (('parent',m,parent['activation_state']),('ldl',new,state)):
                start=time.perf_counter();pa=mod.hif4_dynamic_quantize_activation(*packed,st);r['seconds'][name]=time.perf_counter()-start;assert not validate_hif4_params(pa,x.shape,name)
                z=m._linear_score_decode(pa);r['scores'][name]=1-float((y-z@b.T).square().mean())/bm
            r['delta']=r['scores']['ldl']-r['scores']['parent'];rows.append(r)
        print(gi,sum(r['delta'] for r in rows),flush=True)
    (ROOT/a.output).write_text(json.dumps(dict(complete=True,rows=rows,delta=sum(r['delta'] for r in rows),score_prediction_allowed=False),indent=2)+'\n')
if __name__=='__main__':main()
