"""Math, calibration-cache equivalence, isolation and deployed-grid checks."""
import ast
import copy
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_frozen_state,validate_hif4_params
from build_v320_candidate import candidate_source


def same(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def main():
    torch.set_num_threads(4)
    m=load_solution(ROOT/'solution/solution_v319.py')
    source=candidate_source();exec(compile(source,'v320-test','exec'),m.__dict__)
    banned=set('os pathlib io tempfile shutil subprocess socket ctypes pickle marshal importlib zipimport pkgutil runpy mmap resource pty tty sys builtins'.split())
    for node in ast.walk(ast.parse(source)):
        if isinstance(node,ast.Import):assert not any(x.name.split('.')[0] in banned for x in node.names)
        if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] not in banned
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):assert node.func.id not in ('open','exec','eval','compile','__import__')
    old=ast.parse((ROOT/'solution/solution_v319.py').read_text()).body
    new=ast.parse(source).body
    changed=[]
    for i,node in enumerate(old):
        if ast.dump(node)!=ast.dump(new[i]):
            assert i==0 or isinstance(node,ast.FunctionDef) and node.name in ('_temp_fit','hif4_calibration_attention')
            changed.append(getattr(node,'name','header'))
    assert changed==['header','_temp_fit','hif4_calibration_attention']
    gen=torch.Generator().manual_seed(320)
    # Empirical second moments, not the squared average design; exact feature risk.
    aa=torch.zeros(7,7,dtype=torch.float64);ab=torch.zeros(7,3,dtype=torch.float64)
    samples=[]
    for _ in range(4):
        a=torch.randn(11,7,generator=gen,dtype=torch.float64)
        r=torch.randn(11,3,generator=gen,dtype=torch.float64)
        aa+=a.T@a;ab-=a.T@r;samples.append((a,r))
    ridge=.1;beta=torch.linalg.solve(aa+ridge*torch.eye(7),ab)
    stationary=float(((aa+ridge*torch.eye(7))@beta-ab).abs().max())
    delta=sum(float((a@beta+r).square().sum()-r.square().sum()) for a,r in samples)
    formula=float((beta*(aa@beta)).sum()-2*(beta*ab).sum())
    assert stationary<1e-12 and abs(delta-formula)<1e-12 and delta<0
    p=torch.softmax(torch.randn(9,17,generator=gen,dtype=torch.float64),-1)
    v=torch.randn(17,7,generator=gen,dtype=torch.float64)
    center=v.mean(0);scale=v.square().mean(0).sqrt()
    features=torch.cat(((v-center)/scale,torch.ones(17,1)),1)
    output_features=torch.cat(((p@v-center)/scale,torch.ones(9,1)),1)
    commute_error=float((p@features-output_features).abs().max())
    assert commute_error<1e-12
    # Forced nonzero branch, inactive heads, boundary rows and input/state preservation.
    returns=0;kh,d=2,64
    st=dict(center=torch.zeros(kh,d),rms=torch.ones(kh,d),beta=torch.randn(kh,d+1,d,generator=gen)*.005,
            gain=torch.tensor([1.,0.]),fallback=None)
    assert not validate_frozen_state(dict(vr=st),'new-state')
    changed_outputs=0
    for rows in (1,17,65,129,1025,4097):
        pair=quantize_nvfp4(torch.randn(rows,kh*d,generator=gen)*.3)
        snapshot=copy.deepcopy((pair,st));rng=torch.get_rng_state().clone()
        parent=m._vr_parent_dynamic(*pair,kh,d,None)
        out=m.hif4_dynamic_quantize_v(*pair,kh,d,dict(vr=st))
        assert not validate_hif4_params(out,pair[0].shape,'v320')
        assert same(snapshot,(pair,st)) and torch.equal(rng,torch.get_rng_state())
        assert all(torch.equal(out[k][:,1:],parent[k][:,1:]) for k in out)
        if rows in (1,4097):assert same(out,parent)
        else:changed_outputs+=int(not same(out,parent))
        off=dict(st,gain=torch.zeros(kh))
        assert same(parent,m.hif4_dynamic_quantize_v(*pair,kh,d,dict(vr=off)))
        returns+=1
    assert changed_outputs
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    results=[]
    for gi in (0,1,19):
        g=bank[gi];qh,kh,d=(g[k] for k in ('q_num_heads','kv_num_heads','head_dim'))
        with torch.inference_mode():
            parent=m._vr_parent_calibration(g['calib'],qh,kh,d)
            snapshot=copy.deepcopy(parent)
            cached=m._vr_fit(g['calib'],parent,qh,kh,d)
            plain={k:v for k,v in parent.items() if k!='_vr_cache'}
            fresh=m._vr_fit(g['calib'],plain,qh,kh,d)
            assert same(cached,fresh),(gi,'cache differs')
            altered=list(g['calib']);altered[-1]=altered[0]
            assert same(cached,m._vr_fit(altered,parent,qh,kh,d))
            final=m._vr_guard(g['calib'],parent,qh,kh,d)
            final={k:v for k,v in final.items() if k!='_vr_cache'}
        assert same(parent,snapshot)
        assert not validate_frozen_state(final,'final')
        assert same(final['q_state'],parent['q_state']) and same(final['k_state'],parent['k_state'])
        independent=load_solution(ROOT/'solution/solution_v319.py')
        exec(compile(source,'independent-module','exec'),independent.__dict__)
        for s in g['test']:
            out=m.hif4_dynamic_quantize_v(*s['v'],kh,d,final['v_state'])
            separate=independent.hif4_dynamic_quantize_v(*s['v'],kh,d,copy.deepcopy(final['v_state']))
            assert same(out,separate)
            assert not validate_hif4_params(out,s['v'][0].shape,'real-v320')
            returns+=1
        results.append(dict(group=gi,cache_entries=len(parent.get('_vr_cache',[])),adopted=m.VR_DEBUG.get('accepted')))
    report=dict(passed=True,returns=returns,changed_ast_nodes=changed,stationary_max_error=stationary,
        risk_formula_error=abs(delta-formula),output_map_commute_error=commute_error,cache_equal=True,heldout_excluded_from_fit=True,
        qk_state_unchanged=True,independent_module_equal=True,boundary_rows=[1,17,65,129,1025,4097],groups=results)
    (ROOT/'docs/v320-invariants.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
