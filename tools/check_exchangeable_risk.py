"""Verify the exchangeable probability Gram by exhaustive small permutations."""
import itertools,json
from pathlib import Path
import torch


def main():
    rows=[]
    for values in ((.5,.5),(.1,.9),(.05,.15,.3,.5),(.25,.25,.25,.25),(0.,0.,0.,1.)):
        p=torch.tensor(values,dtype=torch.float64);t=p.numel();s=p.square().sum()
        permutations=torch.stack([p[list(ix)] for ix in itertools.permutations(range(t))])
        h=permutations.T@permutations/permutations.shape[0]
        alpha=(1-s)/(t*(t-1));beta=(t*s-1)/(t*(t-1))
        closed=alpha*torch.ones(t,t,dtype=torch.float64)+beta*torch.eye(t,dtype=torch.float64)
        error=float((h-closed).abs().max())
        assert torch.allclose(h,closed,atol=1e-14,rtol=0)
        assert bool((torch.linalg.eigvalsh(closed)>=-1e-14).all())
        assert torch.allclose(closed.sum(1),torch.full((t,),1/t,dtype=torch.float64))
        penalty=None
        if alpha>0:
            penalty=float((t*s-1)/(1-s))
            e=torch.linspace(-.3,.7,t,dtype=torch.float64)
            proxy=alpha*(e.sum().square()+penalty*e.square().sum())
            assert torch.allclose(e@h@e,proxy,atol=1e-14,rtol=0)
        rows.append(dict(probabilities=values,concentration=float(s),max_error=error,penalty=penalty))
    path=Path(__file__).resolve().parents[1]/'docs/exchangeable-risk-math.json'
    path.write_text(json.dumps(dict(complete=True,rows=rows),indent=2)+'\n')
    print(rows)


if __name__=='__main__':main()
