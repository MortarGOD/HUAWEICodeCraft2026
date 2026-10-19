"""Synthetic active-head solver timing, never an online runtime estimate."""
import json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    m=load_solution(ROOT/'solution/solution_v419.py')
    gen=torch.Generator().manual_seed(419)
    rows=[]
    for t,r in ((2048,16),(2048,64),(4096,128)):
        kh,d=2,256
        x=torch.randn(t,kh*d,generator=gen)*.3
        params=m._dyn_table(x,None,False)
        p=torch.softmax(torch.randn(kh,r,t,generator=gen)*2,-1)
        m._vk_probabilities=lambda *a:p
        state=dict(active=torch.ones(kh,dtype=torch.bool),kernel=torch.zeros(1))
        def run(new):
            result=m._vk_refine(x,params,state,kh,d)
            if new:result=m._rb_refine(x,result,state,kh,d)
            return result
        for new in (False,True):run(new)
        seconds={'412':[],'419':[]}
        for trial in range(4):
            for new in ((False,True) if trial%2==0 else (True,False)):
                start=time.perf_counter();result=run(new)
                seconds['419' if new else '412'].append(time.perf_counter()-start)
        old,new=run(False),run(True)
        rows.append(dict(t=t,probability_rows=r,kv_heads=kh,d=d,seconds=seconds,
                         changed=int((m._deq_params(old)!=m._deq_params(new)).sum())))
    report=dict(complete=True,rows=rows,warning='Synthetic P, four CPU threads, active heads. No fitting, decoding or actual online distribution. Not total submission timing.')
    (ROOT/'docs/v419-solver-profile.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
