"""Fixed-U reconstruction alternatives, measured against a double solve."""
import copy,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
torch.set_num_threads(4)
g=torch.load(ROOT/'local_fitted_archive/attention_feedback_v2_before_structural_20260906/attn.pt',weights_only=False)[47]
m=load_solution(ROOT/'solution/solution_v400.py');args=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
p=m._joint_parent_calibration(g['calib'],*args);u=p['k_state']['u']
snapshot=u.clone();eye=torch.eye(u.shape[-1]).expand(u.shape)
def solve(a):
    inv=torch.linalg.solve_triangular(a,eye.to(a.dtype),upper=True)
    return inv@inv.transpose(-1,-2)
ref=solve(u.double())
functions=dict(original=lambda a:torch.cholesky_inverse(a,upper=True),
               contiguous=lambda a:torch.cholesky_inverse(a.contiguous(),upper=True),
               triangular=solve,double_triangular=lambda a:solve(a.double()).float())
report=dict(torch_version=torch.__version__,shape=list(u.shape),stride=list(u.stride()),dtype=str(u.dtype),trials=64,results={})
for name,fn in functions.items():
    errors=[];started=time.perf_counter()
    for _ in range(64):
        a=u.clone().float();out=fn(a)
        assert torch.equal(u,snapshot) and torch.equal(a,u.float())
        errors.append(float((out.double()-ref).abs().max()/ref.abs().max()))
    report['results'][name]=dict(relative_max_errors=errors,seconds=time.perf_counter()-started,
                               min_error=min(errors),max_error=max(errors),bad=sum(e>1e-3 for e in errors))
(ROOT/'docs/v402-metric-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:{a:b for a,b in v.items() if a!='relative_max_errors'} for k,v in report['results'].items()},indent=2))
