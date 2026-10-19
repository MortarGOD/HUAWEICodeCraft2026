"""Exact Softmax gauge, static source audit, mixed-head and isolation checks."""
import ast
import copy
import json
import torch
from probe_v319_value import ROOT,load_solution
from build_v321_candidate import candidate_source
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_frozen_state,validate_hif4_params
from test_v320_residual import same


def main():
    torch.set_num_threads(4);g=torch.Generator().manual_seed(321)
    source=candidate_source();old=ast.parse((ROOT/'solution/solution_v320.py').read_text()).body;new=ast.parse(source).body
    assert all(ast.dump(a)==ast.dump(b) for a,b in zip(old[1:],new[1:len(old)]))
    banned=set('os pathlib io tempfile shutil subprocess socket ctypes pickle marshal importlib zipimport pkgutil runpy mmap resource pty tty sys builtins'.split())
    for node in ast.walk(ast.parse(source)):
        if isinstance(node,ast.Import):assert all(n.name.split('.')[0] not in banned for n in node.names)
        if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] not in banned
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):assert node.func.id not in ('open','exec','eval','compile','__import__')
    q=torch.randn(11,9,generator=g,dtype=torch.float64);k=torch.randn(17,9,generator=g,dtype=torch.float64)
    center=torch.randn(1,9,generator=g,dtype=torch.float64)
    gauge_error=float((torch.softmax(q@k.T,-1)-torch.softmax(q@(k-center).T,-1)).abs().max());assert gauge_error<1e-12
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False);sample=bank[47]
    qh,kh,d=[sample[key] for key in ('q_num_heads','kv_num_heads','head_dim')]
    m=load_solution(ROOT/'solution/solution_v320.py');exec(compile(source,'key-translation-test','exec'),m.__dict__)
    parent=m._kt_parent_calibration(sample['calib'],qh,kh,d);saved=copy.deepcopy(parent)
    final=m._kt_guard(sample['calib'],parent,qh,kh,d)
    assert same(parent,saved) and same(final['q_state'],parent['q_state']) and same(final['v_state'],parent['v_state'])
    assert not validate_frozen_state(final,'final')
    other=load_solution(ROOT/'solution/solution_v320.py');exec(compile(source,'independent-key-module','exec'),other.__dict__)
    checks=0
    for mode in (2,4):
        st=dict(parent['k_state'],kt_mode=mode,kt_active=torch.arange(kh)%2==0)
        if mode==4:st['kt_probe']=torch.randn(kh,16,d,generator=g)/d**.5
        for rows in (1,17,129,1025,4097):
            pair=quantize_nvfp4(torch.randn(rows,kh*d,generator=g)*.3)
            snap=copy.deepcopy((pair,st));rng=torch.get_rng_state().clone()
            out=m.hif4_dynamic_quantize_k(*pair,kh,d,st)
            assert not validate_hif4_params(out,pair[0].shape,'key')
            assert same(snap,(pair,st)) and torch.equal(rng,torch.get_rng_state())
            assert same(out,other.hif4_dynamic_quantize_k(*pair,kh,d,copy.deepcopy(st)))
            baseline=m._kt_parent_k(*pair,kh,d,parent['k_state'])
            for key,value in out.items():
                inactive=(~st['kt_active']).repeat_interleave(d//64)
                assert torch.equal(value[:,inactive],baseline[key][:,inactive]),(mode,rows,key)
            if rows in (1,4097):assert same(out,baseline)
            checks+=1
    report=dict(passed=True,interface_checks=checks,gauge_max_error=gauge_error,parent_ast_prefix_equal=True,
                source_security_audit=True,inputs_state_rng_unchanged=True,inactive_heads_exact=True,independent_module_exact=True,
                calibration_parent_unchanged=True,qv_states_unchanged=True,accepted=m.KT_DEBUG['accepted'])
    (ROOT/'docs/v321-key-invariants.json').write_text(json.dumps(report,indent=2)+'\n');print(report)


if __name__=='__main__':main()
