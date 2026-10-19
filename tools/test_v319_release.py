"""Release-source isolation, sparse-head equivalence and public-format checks."""
import ast
import copy
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from self_check import validate_hif4_params,validate_frozen_state
from build_v319_candidate import candidate_source
from probe_v319_value import candidate_source as research_source


def main():
    torch.set_num_threads(4)
    source=candidate_source()
    parent=load_solution(ROOT/'solution/solution_v318.py')
    proto=load_solution(ROOT/'solution/solution_v318.py')
    release=load_solution(ROOT/'solution/solution_v318.py')
    exec(compile(research_source(),'value-prototype','exec'),proto.__dict__)
    exec(compile(source,'v319-static-candidate','exec'),release.__dict__)
    proto._VM_INNER_ROUNDS=release._VM_INNER_ROUNDS
    # Same nearby-scale algorithm, but the reference keeps the original
    # full-head dynamic wrapper; compare sparse execution, not a new search.
    for node in ast.parse(source).body:
        if isinstance(node,ast.FunctionDef) and node.name in ('_vm_proposal','_vm_neighbor_scales','_vm_tree_at_sf'):
            exec(compile(ast.get_source_segment(source,node),'nearby-reference','exec'),proto.__dict__)
    banned=set('os pathlib io tempfile shutil subprocess socket ctypes pickle marshal importlib zipimport pkgutil runpy mmap resource pty tty sys builtins'.split())
    for node in ast.walk(ast.parse(source)):
        if isinstance(node,ast.Import):assert not any(a.name.split('.')[0] in banned for a in node.names)
        if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] not in banned
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):assert node.func.id not in ('open','exec','eval','compile','__import__')
    original=ast.parse((ROOT/'solution/solution_v318.py').read_text())
    candidate=ast.parse(source)
    for i,node in enumerate(original.body):
        if i==0 or isinstance(node,ast.FunctionDef) and node.name in ('_quant_chunk','_quant_chunk_vec'):
            continue
        assert ast.dump(node)==ast.dump(candidate.body[i])
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    checks=0;result=[]
    generator=torch.Generator().manual_seed(319)
    xb=torch.randn(17,4,8,2,4,generator=generator)
    for specification in parent.CAND_GRID:
        p=parent._quant_chunk(xb,torch.ones_like(xb),(specification,))
        r=release._vm_tree_at_sf(xb.reshape(17,-1),p['scale_factor'])
        assert all(torch.equal(p[k],r[k]) for k in p)
    for gi in (0,1,4,47,49):
        g=bank[gi];qh,kh,d=(g[k] for k in ('q_num_heads','kv_num_heads','head_dim'))
        st=parent.hif4_calibration_attention(g['calib'],qh,kh,d)
        snapshot=copy.deepcopy(st)
        mask=torch.arange(kh)%2==0
        vstate=dict(vm_active=mask,vm_variance=8.0,fallback=st['v_state'])
        assert not validate_frozen_state(vstate,'value-state')
        for sample in g['test']:
            # Tests in the evaluation bank use the same role tuple layout.
            p=proto.hif4_dynamic_quantize_v(*sample['v'],kh,d,vstate)
            r=release.hif4_dynamic_quantize_v(*sample['v'],kh,d,vstate)
            assert not validate_hif4_params(r,sample['v'][0].shape,'release-value')
            assert all(torch.equal(p[k],r[k]) for k in p),(gi,sample['v'][0].shape)
            checks+=1
            for role,heads in (('q',qh),('k',kh)):
                p=getattr(parent,'hif4_dynamic_quantize_'+role)(*sample[role],heads,d,st[role+'_state'])
                r=getattr(release,'hif4_dynamic_quantize_'+role)(*sample[role],heads,d,st[role+'_state'])
                assert not validate_hif4_params(r,sample[role][0].shape,'release-'+role)
                assert all(torch.equal(p[key],r[key]) for key in p),(gi,role,sample[role][0].shape)
                checks+=1
        for role in ('q_state','k_state'):
            assert all(torch.equal(v,st[role][k]) if isinstance(v,torch.Tensor) else v==st[role][k] for k,v in snapshot[role].items())
        result.append(dict(group=gi,kv_heads=kh,active=int(mask.sum())))
    report=dict(passed=True,returns=checks,groups=result,static_import_scan=True,linear_original_ast_unchanged=True)
    (ROOT/'docs/v319-release-invariants.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__=='__main__':main()
