"""Mathematical and interface invariants; synthetic inputs test correctness only."""
import ast
import copy
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from probe_v318_metadata import configured,candidate_source
from evaluate_local import load_solution
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_hif4_params,validate_frozen_state


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    torch.set_num_threads(4)
    m=configured();parent=load_solution(ROOT/'solution/solution_v317.py')
    rng=torch.Generator().manual_seed(318)
    checks=0; reductions=[]
    for heads,kvh,d,rows in ((1,1,64,19),(4,2,128,33),(4,1,256,129)):
        f=torch.randn(kvh,d,d,generator=rng)
        h=f@f.transpose(-1,-2)+.1*torch.eye(d)[None]
        state=dict(gq=1,rf=1,kvh=kvh,H=h.bfloat16(),u=m._upper_cholesky_inv(h))
        st=m._meta_prepare(state,d);assert st is not None
        assert not validate_frozen_state(st,'prepared')
        x=torch.randn(rows,heads*d,generator=rng)
        p=m._dyn_table(x,None,False);p0=copy.deepcopy(p);x0=x.clone();s0=copy.deepcopy(st)
        out=m._meta_refine(x,p,st,heads,d,rows)
        assert not validate_hif4_params(out,x.shape,'metadata')
        def loss(params):
            error=(m._deq_params(params)-x).reshape(rows,heads,d).transpose(0,1).double()
            hh=st['meta_h'].repeat_interleave(heads//kvh,0).double()
            return ((error@hh)*error).sum(-1)
        old,new=loss(p),loss(out)
        assert bool((new<=old+1e-5*old.clamp_min(1)).all()), float((new-old).max())
        reductions.append(float((old-new).sum()))
        assert equal(p,p0) and torch.equal(x,x0) and equal(st,s0)
        assert equal(p,m._meta_refine(x,p,dict(st,meta_active=torch.zeros(heads,dtype=torch.bool)),heads,d,rows))
        assert equal(p,m._meta_refine(x,p,st,heads,d,1))
        assert equal(p,m._meta_refine(x,p,st,heads,d,4097))
        packed=quantize_nvfp4(x)
        for role in 'qk':
            fn=getattr(m,'hif4_dynamic_quantize_'+role)
            oldfn=getattr(parent,'hif4_dynamic_quantize_'+role)
            assert equal(fn(*packed,heads,d,state),oldfn(*packed,heads,d,state))
            out=fn(*packed,heads,d,st)
            assert not validate_hif4_params(out,x.shape,role)
            checks+=1
        # Query subset must agree with full query execution under original T.
        ix=torch.arange(0,rows,3)
        full=m._deq_params(m.hif4_dynamic_quantize_q(*packed,heads,d,st))[ix]
        sub=m._joint_quant_queries({'q':packed},st,heads,kvh,d,ix)
        assert torch.equal(full,sub), float((full-sub).abs().max())
        checks+=1
    # Incremental quadratic identity in float64, including cross-block terms.
    e=torch.randn(7,128,generator=rng).double();f=torch.randn(128,128,generator=rng).double();h=f@f.T
    delta=torch.zeros_like(e);delta[:,:64]=torch.randn(7,64,generator=rng)
    explicit=(((e+delta)@h)*(e+delta)).sum(-1)-((e@h)*e).sum(-1)
    incremental=(2*delta*(e@h)+(delta@h)*delta).sum(-1)
    assert torch.allclose(explicit,incremental,rtol=1e-10,atol=1e-9)
    before=ast.parse((ROOT/'solution/solution_v317.py').read_text())
    after=ast.parse(candidate_source())
    changed={'_dyn_qk','_joint_quant_queries'}
    for a,b in zip(before.body,after.body):
        if isinstance(a,ast.FunctionDef) and a.name in changed:continue
        assert ast.dump(a)==ast.dump(b),getattr(a,'name',type(a).__name__)
    banned={'os','pathlib','io','tempfile','shutil','subprocess','socket','ctypes','pickle','marshal',
            'importlib','zipimport','pkgutil','runpy','mmap','resource','pty','tty','sys','builtins'}
    for n in ast.walk(after):
        if isinstance(n,ast.Import):assert all(x.name.split('.')[0] not in banned for x in n.names)
        if isinstance(n,ast.ImportFrom):assert (n.module or '').split('.')[0] not in banned
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name):assert n.func.id not in {'exec','eval','open','__import__'}
    report=dict(passed=True,format_and_subset_checks=checks,quadratic_identity=True,
                per_row_quadratic_nonincrease=True,loss_reductions=reductions,
                unchanged_inputs_state_parent=True,linear_ast_equal=True,security_static=True)
    (ROOT/'docs/v318-metadata-invariants.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__=='__main__':main()
