"""Static parent preservation and compiled/research equivalence."""
import ast
import copy
import json
import torch
from probe_v322_translation import ROOT,configured
from build_v322_candidate import candidate_source
from probe_v319_value import load_solution
from test_v320_residual import same
from self_check import validate_hif4_params,validate_attention_calibration_result
from build_fitted_dataset import quantize_nvfp4


def main():
    torch.set_num_threads(4);source=candidate_source()
    parent=(ROOT/'solution/solution_v321.py').read_text()
    old=ast.parse(parent).body;new=ast.parse(source).body
    assert all(ast.dump(a)==ast.dump(b) for a,b in zip(old[1:],new[1:len(old)]) if getattr(a,'name',None)!='_kt_guard')
    banned=set('os pathlib io tempfile shutil subprocess socket ctypes pickle marshal importlib zipimport pkgutil runpy mmap resource pty tty sys builtins'.split())
    for node in ast.walk(ast.parse(source)):
        if isinstance(node,ast.Import):assert all(n.name.split('.')[0] not in banned for n in node.names)
        if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] not in banned
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):assert node.func.id not in ('open','exec','eval','compile','__import__')
    m=load_solution(ROOT/'solution/solution_v321.py');exec(compile(source,'v322','exec'),m.__dict__)
    research=configured();research._TS_MATCH_GUARD=True
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    checks=0;accept=0
    for gi in (4,7,19,47,48,49):
        g=bank[gi];qh,kh,d=[g[x] for x in ('q_num_heads','kv_num_heads','head_dim')]
        base=m._ts_parent_calibration(g['calib'],qh,kh,d);saved=copy.deepcopy(base)
        plain={k:v for k,v in base.items() if k!='_ts_cache'}
        if '_ts_cache' in base:
            exact=research._deq_params(research._ts_parent_k(*g['calib'][-1]['k'],kh,d,plain['k_state']))
            assert torch.equal(exact,base['_ts_cache'][4]),('cached K',gi)
        st=m._ts_guard(g['calib'],base,qh,kh,d);compare=research._ts_guard(g['calib'],plain,qh,kh,d)
        assert same(st,compare) and same(saved,base)
        assert same(st['q_state'],base['q_state']) and same(st['v_state'],base['v_state'])
        assert not validate_attention_calibration_result(st,'attention')
        accept+=m.TS_DEBUG['accepted']
        for sample in g['test']:
            out=m.hif4_dynamic_quantize_k(*sample['k'],kh,d,st['k_state'])
            other=research.hif4_dynamic_quantize_k(*sample['k'],kh,d,copy.deepcopy(st['k_state']))
            assert same(out,other) and not validate_hif4_params(out,sample['k'][0].shape,'k')
            checks+=1
    assert accept>0
    # Exercise a real accepted state through a separately imported module.
    other=load_solution(ROOT/'solution/solution_v321.py');exec(compile(source,'v322-isolated','exec'),other.__dict__)
    g=bank[4];qh,kh,d=[g[x] for x in ('q_num_heads','kv_num_heads','head_dim')]
    final=m.hif4_calibration_attention(g['calib'],qh,kh,d)
    assert '_ts_cache' not in final and 'ts_variant' in final['k_state']
    mixed=dict(final['k_state'],ts_active=torch.arange(kh)%2==0)
    generator=torch.Generator().manual_seed(322)
    for rows in (1,17,129,1025,4097):
        pair=quantize_nvfp4(torch.randn(rows,kh*d,generator=generator)*.4)
        saved=copy.deepcopy((pair,mixed));rng=torch.get_rng_state().clone()
        out=m.hif4_dynamic_quantize_k(*pair,kh,d,mixed)
        assert same(out,other.hif4_dynamic_quantize_k(*pair,kh,d,copy.deepcopy(mixed)))
        assert same(saved,(pair,mixed)) and torch.equal(rng,torch.get_rng_state())
        assert not validate_hif4_params(out,pair[0].shape,'k')
        baseline=research._ts_parent_k(*pair,kh,d,mixed['ts_fallback'])
        inactive=(~mixed['ts_active']).repeat_interleave(d//64)
        for key,value in out.items():assert torch.equal(value[:,inactive],baseline[key][:,inactive])
        if rows in (1,4097):assert same(out,baseline)
        checks+=1
    report=dict(passed=True,parent_ast_equal_except_guard_instrumentation=True,security_ast=True,checks=checks,
                research_production_equal=True,parent_and_qv_unchanged=True,accepted=accept,
                calibration_cache_removed=True,cached_k_exact=True,independent_modules_exact=True,boundary_and_inactive_exact=True)
    (ROOT/'docs/v322-production-invariants.json').write_text(json.dumps(report,indent=2)+'\n');print(report)


if __name__=='__main__':main()
