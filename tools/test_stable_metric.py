"""v402 reconstruction algebra, layouts and fixed-parent repeat audit."""
import ast,copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
torch.set_num_threads(4);torch.manual_seed(402)
m=load_solution(ROOT/'solution/solution_v402.py')
s=(ROOT/'solution/solution_v402.py').read_text()
assert s.startswith((ROOT/'solution/solution_v400.py').read_text())
for node in ast.walk(ast.parse(s)):
    if isinstance(node,ast.Import):assert all(a.name.split('.')[0] in {'torch','numpy','typing','threading'} for a in node.names)
    if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] in {'__future__','typing','torch','numpy','threading'}
max_error=0.
for d in (16,64,128,256):
    for batch in (1,3):
        a=torch.randn(batch,d,d,dtype=torch.float64)/d**.5
        gram=a@a.transpose(-1,-2)+torch.eye(d)*.5
        u=torch.linalg.cholesky(torch.linalg.inv(gram),upper=True).to(torch.bfloat16)
        ref_inv=torch.linalg.solve_triangular(u.double(),torch.eye(d,dtype=torch.float64).expand_as(u),upper=True)
        ref=ref_inv@ref_inv.transpose(-1,-2)
        ref=ref-ref.diagonal(dim1=-2,dim2=-1).mean(-1)[:,None,None]*(m.GPTQ_DAMP/(1+m.GPTQ_DAMP))*torch.eye(d)
        for z in (u,u.contiguous(),u.float()+torch.tril(torch.randn(batch,d,d),diagonal=-1)):
            snapshot=z.clone();rng=torch.random.get_rng_state().clone()
            out=m._joint_raw_metric({'u':z})
            err=float((out.double()-ref).abs().max()/ref.abs().max());max_error=max(max_error,err)
            assert err<1e-5 and torch.equal(z,snapshot) and torch.equal(rng,torch.random.get_rng_state())
        h=torch.randn(batch,d,d).to(torch.bfloat16)
        assert torch.equal(m._joint_raw_metric({'H':h}),m._mr_parent_raw_metric({'H':h}))
g=torch.load(ROOT/'local_fitted_archive/attention_feedback_v2_before_structural_20260906/attn.pt',weights_only=False)[47]
args=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
p=m._joint_parent_calibration(g['calib'],*args)
rows=[];first=None
for i in range(32):
    b=m._joint_balanced_states(copy.deepcopy(p),*args)
    assert b is not None
    values={r+'/'+k:v.clone() for r in 'qk' for k,v in b[r+'_state'].items() if torch.is_tensor(v)}
    if first is None:first=values
    difference=max(float((values[k].float()-first[k].float()).abs().max()) for k in values)
    rows.append(dict(iteration=i,max_difference=difference))
    assert difference==0
(ROOT/'docs/v402-stable-repeat.json').write_text(json.dumps(dict(complete=True,max_random_relative_error=max_error,rows=rows),indent=2)+'\n')
print('PASS: parent/imports, correct inverse order, 8 SPD shapes, layout/lower-triangle invariance, H unchanged, state/RNG, 32 identical fixed-parent balanced states; max error',max_error)
