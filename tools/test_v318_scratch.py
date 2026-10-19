"""Exact-output checks and paired reverse-order GPTQ timing."""
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution


def main():
    torch.set_num_threads(4)
    parent=load_solution(ROOT/'solution/solution_v317.py')
    m=load_solution(ROOT/'solution/solution_v317.py')
    exec(compile((ROOT/'tools/gptq_scratch_core.py').read_text(),'scratch-research','exec'),m.__dict__)
    rng=torch.Generator().manual_seed(3180);cases=[];checks=0
    for heads,rows,d in ((1,1,64),(2,31,128),(4,128,256),(4,1024,128),(8,1024,128),(4,2048,256)):
        x=torch.randn(heads,rows,d,generator=rng)
        unit=torch.exp2(torch.randint(-8,4,(heads,rows,d),generator=rng).float())*1.25
        f=torch.randn(heads,d,d,generator=rng);h=f@f.transpose(-1,-2)+torch.eye(d)[None]
        u=parent._upper_cholesky_inv(h)
        for kind in ('np','columns'):
            fn='_gptq_quantize_batched_'+kind
            inputs=[x.clone(),unit.clone(),u.clone()]
            a=getattr(parent,fn)(x,unit,u);b=getattr(m,fn)(x,unit,u)
            assert torch.equal(a,b),(heads,rows,d,kind,float((a-b).abs().max()))
            assert all(torch.equal(a,b) for a,b in zip(inputs,(x,unit,u)))
            checks+=1
        times=[]
        for mod in (parent,m,m,parent):
            t=time.perf_counter();mod._gptq_quantize_batched(x,unit,u);times.append(time.perf_counter()-t)
        cases.append(dict(shape=[heads,rows,d],times_parent_new_new_parent=times,
                          ratio=(times[1]+times[2])/(times[0]+times[3])))
    report=dict(passed=True,equality_checks=checks,cases=cases,
                note='Finite FP32 inputs, same units and U; local kernel timing, not online total.')
    (ROOT/'docs/v318-scratch-invariants.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
