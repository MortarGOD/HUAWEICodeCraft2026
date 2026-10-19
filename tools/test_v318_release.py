"""Submission-source audit, exact-query tiling and real-bank paired state checks."""
import ast
import copy
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from probe_v318_release import configured,candidate_source
from evaluate_local import load_solution
from test_v318_metadata import equal
from self_check import validate_frozen_state,validate_hif4_params


def main():
    torch.set_num_threads(4)
    parent=load_solution(ROOT/'solution/solution_v317.py');m=configured()
    rng=torch.Generator().manual_seed(3184);tiling=[]
    for rows,heads,kh,d in ((1,4,2,64),(128,4,2,64),(513,4,2,64),(1024,8,2,128),(2048,4,1,128)):
        q=torch.randn(rows,heads*d,generator=rng);k=torch.randn(rows,kh*d,generator=rng);v=torch.randn(rows,kh*d,generator=rng)
        old=parent._attention_out(q,k,v,heads,kh,d);new=m._attention_out(q,k,v,heads,kh,d)
        delta=float((old-new).abs().max())
        assert torch.allclose(old,new,rtol=1e-5,atol=1e-6),delta
        tiling.append(dict(rows=rows,heads=heads,d=d,max_abs=delta,equal=torch.equal(old,new)))
    old_source=(ROOT/'solution/solution_v317.py').read_text();new_source=candidate_source()
    before=ast.parse(old_source);after=ast.parse(new_source)
    changed={'_gptq_quantize_batched_np','_gptq_quantize_batched_columns','_attention_out','hif4_calibration_attention'}
    assert len(before.body)==len(after.body)
    for a,b in zip(before.body[1:],after.body[1:]):
        if isinstance(a,ast.FunctionDef) and a.name in changed:continue
        assert ast.dump(a)==ast.dump(b),getattr(a,'name',type(a).__name__)
    for name in ('hif4_calibration_and_quantize_weight','hif4_dynamic_quantize_activation'):
        a=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a)==ast.dump(b)
    banned={'os','pathlib','io','tempfile','shutil','subprocess','socket','ctypes','pickle','marshal','importlib',
            'zipimport','pkgutil','runpy','mmap','resource','pty','tty','sys','builtins'}
    for n in ast.walk(after):
        if isinstance(n,ast.Import):assert all(x.name.split('.')[0] not in banned for x in n.names)
        if isinstance(n,ast.ImportFrom):assert (n.module or '').split('.')[0] not in banned
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name):assert n.func.id not in {'exec','eval','open','__import__'}
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    checks=0;paired=[]
    for gi in (1,5,19,47,49):
        g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        a=parent.hif4_calibration_attention(g['calib'],qh,kh,d);b=m.hif4_calibration_attention(g['calib'],qh,kh,d)
        assert not validate_frozen_state(b,'release')
        state_same=equal(a,b);same_returns=True
        for sample in g['test']:
            for role,heads in (('q',qh),('k',kh),('v',kh)):
                original=getattr(parent,'hif4_dynamic_quantize_'+role)(*sample[role],heads,d,a[role+'_state'])
                fixed=getattr(m,'hif4_dynamic_quantize_'+role)(*sample[role],heads,d,a[role+'_state'])
                actual=getattr(m,'hif4_dynamic_quantize_'+role)(*sample[role],heads,d,b[role+'_state'])
                assert equal(original,fixed),('fixed state',gi,role)
                assert not validate_hif4_params(actual,sample[role][0].shape,role)
                same_returns=same_returns and equal(original,actual);checks+=1
        paired.append(dict(group=gi,calibration_state_equal=state_same,all_returns_equal=same_returns))
    report=dict(passed=True,linear_ast_equal=True,other_functions_ast_equal=True,security_static=True,
                tiling=tiling,paired=paired,format_checks=checks,
                warning='If independently recomputed states differ, retain that evidence; fixed-state dynamics must match. Tiling is mathematically exact, floating-point equality checked only for these shapes.')
    (ROOT/'docs/v318-release-invariants.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
