"""Fixed-grid channel-wise translation phase candidates; not a submission."""
import json
from pathlib import Path
import torch


def phase_search(y,center,step,weight,count=17):
    scale=(step*weight).sum(0,keepdim=True)
    def cost(c):
        z=y-c
        error=(z/step).round().clamp(-7,7)*step-z
        return (error.square()*weight).sum(0,keepdim=True)
    best_center=center.clone();best=cost(center)
    for phase in torch.linspace(-1.,1.,count,dtype=y.dtype,device=y.device):
        trial=center+phase*scale
        candidate=cost(trial);take=candidate<best
        best_center=torch.where(take,trial,best_center)
        best=torch.minimum(best,candidate)
    return best_center,best


def main():
    generator=torch.Generator().manual_seed(352);checks=0;improved=0;relative=[]
    for seed in range(100):
        y=torch.randn(17,2,7,generator=generator,dtype=torch.float64)
        step=torch.randint(-3,2,y.shape,generator=generator).to(y.dtype).exp2()
        weight=torch.randn(17,2,1,generator=generator,dtype=y.dtype).softmax(0)
        original=y.clone();center=(y*weight).sum(0,keepdim=True)
        for _ in range(8):
            z=y-center;error=(z/step).round().clamp(-7,7)*step-z
            center=center-(error*weight).sum(0,keepdim=True)
        z=y-center;old=((z/step).round().clamp(-7,7)*step-z).square().mul(weight).sum(0,keepdim=True)
        new,loss=phase_search(y,center,step,weight)
        assert bool((loss<=old+1e-14).all())
        assert torch.equal(y,original) and torch.isfinite(new).all()
        q=torch.randn(2,5,7,generator=generator,dtype=y.dtype)
        a=(q@y.transpose(0,1).transpose(1,2)).softmax(-1)
        b=(q@(y-new).transpose(0,1).transpose(1,2)).softmax(-1)
        assert torch.allclose(a,b,atol=2e-14,rtol=0)
        result=((y-new)/step).round().clamp(-7,7)*step
        assert torch.allclose(result/step,(result/step).round(),atol=1e-13,rtol=0)
        improved+=int(bool((loss<old-1e-12).any()))
        relative.append(float(loss.sum()/old.sum()));checks+=4
    report=dict(checks=checks,random_instances=100,instances_with_improved_channel=improved,
                mean_fixed_grid_objective_ratio=sum(relative)/len(relative),
                warning='Toy fixed-scale surrogate only. No real-data or online gain; recomputing scales/GPTQ can reverse the improvement.')
    path=Path(__file__).resolve().parents[1]/'docs/key-phase-math.json'
    path.write_text(json.dumps(report,indent=2)+'\n');print(report)


if __name__=='__main__':main()
