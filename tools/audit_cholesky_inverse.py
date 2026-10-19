"""Independent CPU numerical-contract audit, not a contestant modification."""
import json
import argparse
from pathlib import Path
import platform
import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--output',default='docs/cholesky-inverse-local-audit.json')
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    rng=torch.Generator().manual_seed(316101)
    results=[]
    for dtype in (torch.float32,torch.float64):
        for d in (64,128,256,512):
            for batch in (1,2):
                u=(torch.randn(batch,d,d,dtype=dtype,generator=rng)*.01).triu(1)+torch.eye(d,dtype=dtype)
                identity=torch.eye(d,dtype=torch.float64).expand(batch,d,d)
                iu=torch.linalg.solve_triangular(u.double(),identity,upper=True)
                safe=iu@iu.transpose(-1,-2)
                inverse_source=u.double().transpose(-1,-2)@u.double()
                safe_residual=float((inverse_source@safe-identity).abs().max())
                assert safe_residual<1e-10,safe_residual
                for upper in (False,True):
                    factor=u if upper else u.transpose(-1,-2).contiguous()
                    residuals=[]
                    errors=[]
                    for _ in range(5):
                        result=torch.cholesky_inverse(factor,upper=upper).double()
                        residuals.append(float((inverse_source@result-identity).abs().max()))
                        errors.append(float((result-safe).abs().max()))
                    bound=1e-4 if dtype==torch.float32 else 1e-10
                    results.append(dict(dtype=str(dtype),d=d,batch=batch,upper=upper,
                                        safe_residual=safe_residual,
                                        residuals=residuals,errors=errors,
                                        failures=sum(x>bound for x in residuals)))
    report=dict(torch=torch.__version__,platform=platform.platform(),threads=torch.get_num_threads(),
                config=torch.__config__.show(),results=results,
                failures=sum(x['failures'] for x in results),calls=5*len(results),
                note='Tests the local numerical contract only; not evidence of the online judge backend.')
    root=Path(__file__).resolve().parents[1]
    (root/args.output).write_text(json.dumps(report,indent=2)+'\n')
    print('failures',report['failures'],'/',report['calls'])
    for r in results:
        if r['failures']:print(r)


if __name__=='__main__':main()
