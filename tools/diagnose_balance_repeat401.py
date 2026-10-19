"""Bounded identical-parent balance decomposition audit; no submission changes."""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
torch.set_num_threads(4)
g=torch.load(ROOT/'local_fitted_archive/attention_feedback_v2_before_structural_20260906/attn.pt',weights_only=False)[47]
m=load_solution(ROOT/'solution/solution_v400.py');args=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
parent=m._joint_parent_calibration(g['calib'],*args)
rows=[];first={};counter=0
raw_first={};raw_parent=m._joint_raw_metric
def traced_raw(st):
    role='q' if 'H' in st else 'k'
    out=raw_parent(st)
    if role not in raw_first:raw_first[role]=out.clone()
    rows.append(dict(stage='raw_metric',role=role,maxdiff=float((out-raw_first[role]).abs().max())))
    return out
m._joint_raw_metric=traced_raw
def traced(h):
    global counter
    role='q' if counter%2==0 else 'k';counter+=1
    hd=h+torch.eye(h.shape[-1])*h.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-30)[:,None,None]*m.GPTQ_DAMP
    if role not in first:first[role]=h.clone()
    row=dict(role=role,input_maxdiff=float((h-first[role]).abs().max()))
    try:
        chol=torch.linalg.cholesky(hd)
        inverse=torch.cholesky_inverse(chol)
        row['inverse_symmetry']=float((inverse-inverse.transpose(-1,-2)).abs().max())
        out=torch.linalg.cholesky(inverse,upper=True)
        row['status']='pass';return out
    except RuntimeError as e:
        row['status']=str(e)
        try:
            c=torch.linalg.cholesky(hd.double());inv=torch.cholesky_inverse(c)
            torch.linalg.cholesky(inv,upper=True);row['float64_retry']='pass'
        except RuntimeError as e2:row['float64_retry']=str(e2)
        return None
    finally:rows.append(row)
m._upper_cholesky_inv=traced
for _ in range(12):m._joint_balanced_states(copy.deepcopy(parent),*args)
(ROOT/'docs/v401-balance-repeat.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
