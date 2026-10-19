"""Validate reconstructed damped Gram algebra and preserved H fast path."""
import copy,json
import torch
from probe_v319_value import ROOT,load_solution
from stable_joint_source import triangular_metric_source
from probe_linear_deployed_joint import tree_equal


def main():
    torch.set_num_threads(4);gen=torch.Generator().manual_seed(352)
    m=load_solution(ROOT/'solution/solution_v324.py')
    source=triangular_metric_source((ROOT/'solution/solution_v324.py').read_text())
    begin=source.index('def _joint_raw_metric(');end=source.index('\ndef _joint_balanced_states(',begin)
    exec(compile(source[begin:end],'triangular_metric','exec'),m.__dict__)
    checks=0;residuals=[]
    for d in (4,16,64,256):
        for batch in (1,2):
            for amplitude in (.01,1.,100.):
                r=torch.randn(batch,d,d,generator=gen,dtype=torch.float64)
                h=(r@r.transpose(-1,-2)/d+torch.eye(d)*.2)*amplitude
                ridge=h.diagonal(dim1=-2,dim2=-1).mean(-1)*m.GPTQ_DAMP
                hd=h+ridge[:,None,None]*torch.eye(d)
                u=torch.linalg.cholesky(torch.linalg.inv(hd),upper=True).float()
                state={'u':u};snapshot=copy.deepcopy(state)
                recovered=m._joint_raw_metric(state)
                relative=float((recovered.double()-h).norm()/h.norm());residuals.append(relative)
                assert relative<1e-4
                assert tree_equal(state,snapshot)
                assert recovered.dtype==torch.float32 and torch.isfinite(recovered).all()
                carried=h.to(torch.bfloat16);fast=m._joint_raw_metric({'H':carried,'u':u})
                assert torch.equal(fast,(carried.float()+carried.float().transpose(-1,-2))*.5)
                checks+=4
    report={'checks':checks,'max_relative_error':max(residuals)}
    (ROOT/'docs/triangular-metric-math.json').write_text(json.dumps(report,indent=2)+'\n');print(report)


if __name__=='__main__':main()
