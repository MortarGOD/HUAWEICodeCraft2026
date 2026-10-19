import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from quant_three_levels import rewrite


def main():
    torch.set_num_threads(4)
    parent=load_solution(ROOT/'solution/solution_v318.py')
    candidate=load_solution(ROOT/'solution/solution_v318.py')
    exec(compile(rewrite((ROOT/'solution/solution_v318.py').read_text()),'three-level-reuse','exec'),candidate.__dict__)
    generator=torch.Generator().manual_seed(319)
    report=dict(passed=False,checks=[],timings=[])
    for rows,blocks in ((1,1),(17,4),(65,16),(128,64)):
        x=torch.randn(rows,blocks,8,2,4,generator=generator)
        weight=torch.rand(1,blocks,8,2,4,generator=generator)
        for mode in ('random','ties','zeros','wide'):
            if mode=='random':xb=x
            elif mode=='ties':xb=(x*4).round()*.25
            elif mode=='zeros':xb=torch.zeros_like(x)
            else:xb=x*torch.exp2(torch.linspace(-30,14,blocks))[None,:,None,None,None]
            for function in ('_quant_chunk','_quant_chunk_vec'):
                p=getattr(parent,function)(xb,weight,parent.CAND_GRID)
                c=getattr(candidate,function)(xb,weight,candidate.CAND_GRID)
                assert all(torch.equal(p[k],c[k]) for k in p),(rows,blocks,mode,function)
                report['checks'].append([rows,blocks,mode,function])
    x=torch.randn(256,64,8,2,4,generator=generator);weight=torch.ones(1,64,8,2,4)
    for name,mod in (('parent',parent),('candidate',candidate),('candidate',candidate),('parent',parent)):
        start=time.perf_counter()
        for unused in range(3):mod._quant_chunk_vec(x,weight,mod.CAND_GRID)
        report['timings'].append(dict(mode=name,seconds=time.perf_counter()-start))
    report['passed']=True
    (ROOT/'docs/v319-three-level-invariants.json').write_text(json.dumps(report,indent=2)+'\n')
    print('checks',len(report['checks']),report['timings'])


if __name__=='__main__':main()
