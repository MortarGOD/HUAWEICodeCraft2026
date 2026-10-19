"""Compose coarse/fine legal-tree updates with one original MSE bound and P."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
full=(ROOT/'solution/solution_v433.py').read_text()
small=(ROOT/'solution/solution_v431.py').read_text()
prefix=full[:full.index('\ndef _a433_solve_nodes')]
coarse=full[full.index('\ndef _a433_solve_nodes'):full.index('\ndef hif4_dynamic_quantize_v(vq,vs,kh,d,state):')]
coarse=coarse.replace('def _a433_tree(x,params,state,kh,d):','def _a433_tree(x,params,state,kh,d,probability=None):')
coarse=coarse.replace('    probability=_ac_parent_probability(x,state,kh)','    if probability is None:probability=_ac_parent_probability(x,state,kh)')
coarse=coarse.replace('a433_active','a434_active')
fine=small[small.index('\ndef _a431_tree'):small.index('\ndef hif4_dynamic_quantize_v(vq,vs,kh,d,state):')]
fine=fine.replace('def _a431_tree(x,params,state,kh,d):','def _a434_fine(x,params,state,kh,d,probability,original_cap):')
fine=fine.replace('    probability=_ac_parent_probability(x,state,kh)\n','')
fine=fine.replace('a431_active','a434_active').replace('(new_mse<=cap[:,None]*1.05)','(new_mse<=original_cap[:,None]*1.05)')
deployment='''

def _a434_hierarchy(x,params,state,kh,d):
    t=x.shape[0]
    if not 2<=t<=4096 or d%64 or t*kh*d>2**23:return params
    probability=_ac_parent_probability(x,state,kh)
    if probability.shape[1]>128 or not bool(torch.isfinite(probability).all()):return params
    error=(_deq_params(params).float()-x).reshape(t,kh,d).transpose(0,1)
    original_cap=error.square().mean(1).clamp_min(1e-30)
    coarse=_a433_tree(x,params,state,kh,d,probability)
    return _a434_fine(x,coarse,state,kh,d,probability,original_cap)


def hif4_dynamic_quantize_v(vq,vs,kh,d,state):
    params=_ac_parent_v(vq,vs,kh,d,state)
    if not isinstance(state,dict) or 'a434_active' not in state:return params
    return _a434_hierarchy(dequantize_nvfp4(vq,vs).float(),params,state,kh,d)
'''
calibration=full[full.rindex('\n@torch.inference_mode()'):].replace('a433_active','a434_active')
result=prefix+coarse+fine+deployment+calibration
ast.parse(result)
(ROOT/'solution/solution_v434.py').write_text(result)
print('Built v434, one P and original MSE cap for both hierarchy levels.')
