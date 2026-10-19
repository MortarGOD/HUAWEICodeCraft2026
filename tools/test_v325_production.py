"""Mathematics, standalone loading, native outputs and frozen-parent checks."""
import ast,copy,hashlib,json
import torch
from build_v325_candidate import ROOT,candidate_source
from probe_v319_value import load_solution
from probe_linear_deployed_joint import tree_equal
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_hif4_params,validate_attention_calibration_result


def main():
    torch.set_num_threads(4);src=candidate_source()
    old=ast.parse((ROOT/'solution/solution_v324.py').read_text()).body;new=ast.parse(src).body
    assert [ast.dump(n) for n in old[1:]]==[ast.dump(n) for n in new[1:len(old)]]
    banned=set('os pathlib io tempfile shutil subprocess socket ctypes pickle marshal importlib zipimport pkgutil runpy mmap resource pty tty sys builtins'.split())
    for n in ast.walk(ast.parse(src)):
        if isinstance(n,ast.Import):assert all(x.name.split('.')[0] not in banned for x in n.names)
        if isinstance(n,ast.ImportFrom):assert n.module.split('.')[0] not in banned
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name):assert n.func.id not in ('open','exec','eval','compile','__import__')
    checks=2;generator=torch.Generator().manual_seed(325)
    a=torch.randn(2,128,128,generator=generator,dtype=torch.float64);h=a@a.transpose(-1,-2)+torch.eye(128)
    e=torch.randn(7,2,128,generator=generator,dtype=torch.float64)
    delta=torch.randn(7,2,64,generator=generator,dtype=torch.float64)*.02
    gradient=torch.einsum('thd,hde->the',e,h)[:,:,:64]
    predicted=2*(gradient*delta).sum(-1)+torch.einsum('thc,hce,the->th',delta,h[:,:64,:64],delta)
    after=e.clone();after[:,:,:64]+=delta
    difference=torch.einsum('thc,hce,the->th',after,h,after)-torch.einsum('thc,hce,the->th',e,h,e)
    maxerr=float((difference-predicted).abs().max());assert maxerr<1e-8;checks+=1
    m=load_solution(ROOT/'solution/solution_v324.py');other=load_solution(ROOT/'solution/solution_v324.py')
    exec(compile(src,'v325-test','exec'),m.__dict__);exec(compile(src,'v325-independent','exec'),other.__dict__)
    for d in (64,128):
        kh,qh=2,4
        a=torch.randn(kh,d,d,generator=generator);h=a@a.transpose(-1,-2)+torch.eye(d)*d
        root=dict(kvh=kh,gq=1,u=m._upper_cholesky_inv(h),H=h.to(torch.bfloat16),rf=1)
        u=root['u'];inverse=torch.linalg.solve_triangular(u,torch.eye(d).expand_as(u),upper=True)
        metric=inverse@inverse.transpose(-1,-2);metric/=metric.diagonal(dim1=-2,dim2=-1).mean(-1)[:,None,None]
        state=dict(root,qg_mode=4,qg_active=torch.tensor([True,False,True,False]),qg_fallback=root,kg_h=metric.repeat_interleave(2,0))
        for t in (1,3,4,17,128,129,1025,4097):
            packed=quantize_nvfp4(torch.randn(t,qh*d,generator=generator)*.4)
            saved=copy.deepcopy((packed,state));rng=torch.random.get_rng_state().clone()
            out=m.hif4_dynamic_quantize_q(*packed,qh,d,state)
            compare=other.hif4_dynamic_quantize_q(*packed,qh,d,copy.deepcopy(state))
            assert tree_equal(out,compare)
            assert not validate_hif4_params(out,packed[0].shape,'q')
            assert tree_equal(saved,(packed,state)) and torch.equal(rng,torch.random.get_rng_state())
            base=m._qg_parent_q(*packed,qh,d,root)
            inactive=(~state['qg_active']).repeat_interleave(d//64)
            assert all(torch.equal(v[:,inactive],base[k][:,inactive]) for k,v in out.items()),('inactive',d,t)
            if t in (1,3,4097):assert tree_equal(out,base)
            checks+=5
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    g=bank[2];dims=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
    root=m._qg_parent_calibration(g['calib'],*dims);saved=copy.deepcopy(root)
    state=m._qg_guard(g['calib'],root,*dims)
    assert not validate_attention_calibration_result(state,'attention')
    assert tree_equal(root,saved) and tree_equal(state['k_state'],root['k_state']) and tree_equal(state['v_state'],root['v_state'])
    assert m.QG_DEBUG['accepted']>0;checks+=3
    report=dict(passed=True,checks=checks,sha256=hashlib.sha256(src.encode()).hexdigest(),quadratic_max_error=maxerr,
        parent_ast_unchanged=True,standalone_equal=True,inputs_and_dynamic_rng_unchanged=True,inactive_exact=True,key_value_state_unchanged=True)
    (ROOT/'docs/v325-invariants.json').write_text(json.dumps(report,indent=2)+'\n');print(report)


if __name__=='__main__':main()
