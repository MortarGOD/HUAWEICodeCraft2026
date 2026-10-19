"""Bounded calibration correctness, long-sample coverage, and frozen Linear."""
import ast
import copy
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from probe_v318_bounded import candidate_source,configured
from evaluate_local import load_solution,gqa_attention
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_frozen_state,validate_hif4_params
from test_v318_metadata import equal


def main():
    torch.set_num_threads(4)
    m=configured();parent=load_solution(ROOT/'solution/solution_v317.py')
    rng=torch.Generator().manual_seed(3183)
    # Slicing Q only preserves softmax; never assert the same for slicing K.
    for qh,kh,d,rows in ((4,2,64,131),(2,1,64,4097)):
        q=torch.randn(rows,qh*d,generator=rng);k=torch.randn(rows,kh*d,generator=rng);v=torch.randn(rows,kh*d,generator=rng)
        ix=torch.linspace(0,rows-1,17).long()
        expected=gqa_attention(q,k,v,qh,kh,d)[ix]
        assert torch.allclose(m._cal_output(q[ix],k,v,qh,kh,d),expected,rtol=1e-5,atol=1e-6)
    # All fitted samples beyond the previous 1024 limit must supply a nonzero H.
    samples=[{r:quantize_nvfp4(torch.randn(t,(2 if r=='q' else 1)*64,generator=rng))
              for r in 'qkv'} for t in (1057,1103,64)]
    snapshot=copy.deepcopy(samples);global_rng=torch.get_rng_state().clone();collected=[]
    original=m._upper_cholesky_inv
    def capture(h):
        collected.append(h.clone());return original(h)
    m._upper_cholesky_inv=capture
    states=m.hif4_calibration_attention(samples,2,1,64)
    assert not validate_frozen_state(states,'long-calibration')
    assert len(collected)>=2 and all(bool(h.diagonal(dim1=-2,dim2=-1).sum()>0) for h in collected[:2])
    assert equal(samples,snapshot) and torch.equal(global_rng,torch.get_rng_state())
    assert not m.TEMP_DEBUG.get('error') and not m.JOINT_DEBUG.get('error')
    checks=0
    for rows in (1,17,129,1024,1025,4097):
        for role,heads in (('q',2),('k',1),('v',1)):
            packed=quantize_nvfp4(torch.randn(rows,heads*64,generator=rng))
            fn=getattr(m,'hif4_dynamic_quantize_'+role)
            out=fn(*packed,heads,64,states[role+'_state'])
            assert not validate_hif4_params(out,packed[0].shape,role)
            # Fixed-state dynamic policies unchanged; scratch only is equivalent.
            old=getattr(parent,'hif4_dynamic_quantize_'+role)(*packed,heads,64,states[role+'_state'])
            assert equal(out,old),(role,rows)
            checks+=1
    before=ast.parse((ROOT/'solution/solution_v317.py').read_text());after=ast.parse(candidate_source())
    changed={'_gptq_quantize_batched_np','_gptq_quantize_batched_columns','hif4_calibration_attention'}
    for a,b in zip(before.body,after.body):
        if isinstance(a,ast.FunctionDef) and a.name in changed:continue
        assert ast.dump(a)==ast.dump(b),getattr(a,'name',type(a).__name__)
    banned={'os','pathlib','io','tempfile','shutil','subprocess','socket','ctypes','pickle','marshal','importlib',
            'zipimport','pkgutil','runpy','mmap','resource','pty','tty','sys','builtins'}
    for n in ast.walk(after):
        if isinstance(n,ast.Import):assert all(x.name.split('.')[0] not in banned for x in n.names)
        if isinstance(n,ast.ImportFrom):assert (n.module or '').split('.')[0] not in banned
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name):assert n.func.id not in {'exec','eval','open','__import__'}
    report=dict(passed=True,format_checks=checks,fixed_state_dynamic_equal=True,long_calibration_nonzero_grams=True,
                query_only_subset_equivalent=True,inputs_rng_unchanged=True,linear_ast_equal=True,security_static=True)
    (ROOT/'docs/v318-bounded-invariants.json').write_text(json.dumps(report,indent=2)+'\n');print(report)


if __name__=='__main__':main()
