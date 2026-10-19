"""Check per-head two-fold selection, fallback and nonfinite rejection."""
from pathlib import Path
import sys
import math
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution


def main():
    m=load_solution(ROOT/'solution/solution_v377.py')
    rng=torch.Generator().manual_seed(377)
    before=torch.rand(2,2000,generator=rng)+.1
    costs=before[None]*(.8+.4*torch.rand(2,2,2000,generator=rng))
    costs[0,0,0]=float('nan');costs[1,0,1]=float('inf');before[:,2]=float('inf')
    costs[:,:,3]=before[:,3]*.999
    costs[:,:,4]=before[:,4]*.9
    choice,take=m._ms_choose(before,costs)
    for h in range(2000):
        valid=[k for k in range(2) if all(math.isfinite(float(before[j,h])) and math.isfinite(float(costs[k,j,h])) and float(costs[k,j,h])<float((before*.995)[j,h]) for j in range(2))]
        assert bool(take[h])==bool(valid)
        if valid:assert int(choice[h])==min(valid,key=lambda k:float(costs[k,:,h].mean()))
    assert not bool(take[2]) and not bool(take[3]) and int(choice[4])==0
    print('PASS:2000 head decisions, two-fold eligibility, ties, no-gain and nonfinite fallback')


if __name__=='__main__':main()
