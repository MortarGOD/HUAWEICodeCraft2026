"""Correctness-only mathematical, source, state and independent-loader checks."""
import ast,copy,hashlib,json
import torch
from probe_v319_value import ROOT,load_solution
from build_v324_candidate import candidate_source
from build_fitted_dataset import quantize_nvfp4
from probe_linear_deployed_joint import tree_equal
from self_check import validate_linear_calibration_result,validate_hif4_params

def main():
    torch.set_num_threads(4);torch.manual_seed(324)
    src=candidate_source();base=(ROOT/'solution/solution_v322.py').read_text();old=ast.parse(base);new=ast.parse(src)
    modified={'_quantize_linear_exact_target','hif4_dynamic_quantize_activation','_linear_refine_fixed_decoder'}
    oldnodes=[n for n in old.body if isinstance(n,ast.FunctionDef)];newnodes=[n for n in new.body if isinstance(n,ast.FunctionDef)]
    assert [ast.dump(n) for n in oldnodes if n.name not in modified]==[ast.dump(n) for n in newnodes[:-1] if n.name not in modified]
    assert newnodes[-1].name=='hif4_calibration_and_quantize_weight'
    forbidden={'os','pathlib','io','tempfile','shutil','subprocess','socket','ctypes','pickle','marshal','importlib','zipimport','pkgutil','runpy','mmap','resource','pty','tty','sys','builtins'}
    for n in ast.walk(new):
        if isinstance(n,ast.Import):assert all(v.name.split('.')[0] not in forbidden for v in n.names)
        if isinstance(n,ast.ImportFrom):assert n.module.split('.')[0] not in forbidden
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name):assert n.func.id not in {'open','exec','eval','compile'}
    checks=2
    b=torch.randn(192,128,dtype=torch.float64);x=torch.randn(7,128,dtype=torch.float64);y=torch.randn(7,192,dtype=torch.float64)
    h=b.T@b;r=y-x@b.T;g=r@b;maxerr=0.
    for begin in (0,64,0,64):
        delta=torch.randn(7,64,dtype=torch.float64)*.01
        oldloss=r.square().sum(-1)
        predicted=2*(delta*g[:,begin:begin+64]).sum(-1)+torch.einsum('rc,cd,rd->r',delta,h[begin:begin+64,begin:begin+64],delta)
        r.add_(delta@b[:,begin:begin+64].T);g.add_(delta@h[begin:begin+64])
        error=float((g-r@b).abs().max());maxerr=max(maxerr,error)
        assert torch.allclose(g,r@b,rtol=1e-10,atol=1e-10)
        assert torch.allclose(r.square().sum(-1)-oldloss,predicted,atol=1e-9,rtol=1e-9);checks+=2
    parent=load_solution(ROOT/'solution/solution_v322.py');m=load_solution(ROOT/'solution/solution_v322.py');independent=load_solution(ROOT/'solution/solution_v322.py')
    exec(compile(src,'candidate','exec'),m.__dict__);exec(compile(src,'independent','exec'),independent.__dict__)
    # Boundary gating without allocating a >8192-wide dense Gram.
    original=m._gram_parent_calibration
    dummy={'activation_state':{'linear_shape':1},'weight_params':{}}
    m._gram_parent_calibration=lambda *args:copy.deepcopy(dummy)
    skipped=m.hif4_calibration_and_quantize_weight(torch.zeros(1).expand(16386,8193),None,[])
    assert 'linear_full_gram' not in skipped['activation_state'];checks+=1
    m._gram_parent_calibration=original
    parent_rng_changes=0
    for outputs,inputs in ((128,64),(256,128),(192,128),(128,128),(64,128)):
        w=quantize_nvfp4(torch.randn(outputs,inputs));cal=[quantize_nvfp4(torch.randn(n,inputs)) for n in (4,8,16,32,64)]
        before=copy.deepcopy((w,cal));rng=torch.random.get_rng_state().clone()
        root=parent.hif4_calibration_and_quantize_weight(*w,cal)
        parent_rng_changes+=int(not torch.equal(rng,torch.random.get_rng_state()))
        after_parent_rng=torch.random.get_rng_state().clone()
        torch.random.set_rng_state(rng)
        m._gram_parent_calibration=original
        actual=m.hif4_calibration_and_quantize_weight(*w,cal)
        actual_cmp=copy.deepcopy(actual);actual_gram=actual_cmp['activation_state'].pop('linear_full_gram',None)
        assert tree_equal(actual_cmp,root), 'Actual decoder or selection changed'
        assert torch.equal(after_parent_rng,torch.random.get_rng_state())
        if actual_gram is not None:
            decoded=m._linear_score_decode(actual['weight_params']).float()
            assert torch.equal(actual_gram,decoded.T@decoded)
        checks+=3
        rng=torch.random.get_rng_state().clone()
        m._gram_parent_calibration=lambda *args,_r=root:copy.deepcopy(_r)
        st=m.hif4_calibration_and_quantize_weight(*w,cal)
        assert not validate_linear_calibration_result(st,(outputs,inputs),'gram')
        cmp=copy.deepcopy(st);gram=cmp['activation_state'].pop('linear_full_gram',None)
        assert tree_equal(cmp,root) and tree_equal((w,cal),before)
        assert torch.equal(rng,torch.random.get_rng_state());checks+=3
        if outputs>=2*inputs:
            bb=m._linear_score_decode(st['weight_params']).float();assert torch.equal(gram,bb.T@bb)
        else:assert gram is None
        checks+=1
        for n in (1,7,128,129,512,513):
            packed=quantize_nvfp4(torch.randn(n,inputs));saved=copy.deepcopy((packed,st));rng=torch.random.get_rng_state().clone()
            params=m.hif4_dynamic_quantize_activation(*packed,st['activation_state'])
            other=independent.hif4_dynamic_quantize_activation(*packed,copy.deepcopy(st['activation_state']))
            assert tree_equal(params,other)
            assert not validate_hif4_params(params,(n,inputs),'dynamic')
            assert tree_equal((packed,st),saved) and torch.equal(rng,torch.random.get_rng_state());checks+=3
    report=dict(passed=True,checks=checks,sha256=hashlib.sha256(src.encode()).hexdigest(),fp64_gradient_max_abs_error=maxerr,unchanged_functions=len(oldnodes)-len(modified),attention_unchanged=True,parent_calibration_rng_changes=parent_rng_changes,warning='Synthetic tensors are correctness-only. The new Gram wrapper adds no RNG changes to the existing parent calibration; parent itself can change RNG. >8192 bound tests fallback, not 8192 cost.')
    (ROOT/'docs/v324-gram-invariants.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':main()
