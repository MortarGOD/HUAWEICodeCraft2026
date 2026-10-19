"""Observe live calibration inverse-factor residuals without altering outputs."""
import argparse
import inspect
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.verify_attention_campaign import role_calibration

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--failures-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);m=load(435,'factor_audit');original=m._upper_cholesky_inv
    records=[];context={};raw_records=[]
    original_raw=m._joint_raw_metric
    def raw_observed(st):
        h=original_raw(st)
        row=dict(context,uses_carried_H='H' in st)
        if 'u' in st:
            u=torch.triu(st['u'].double())
            inverse_u=torch.linalg.solve_triangular(u,torch.eye(u.shape[-1]).double().expand_as(u),upper=True)
            hd=inverse_u@inverse_u.transpose(-1,-2)
            ridge=hd.diagonal(dim1=-2,dim2=-1).mean(-1)*(m.GPTQ_DAMP/(1+m.GPTQ_DAMP))
            recovered=hd-ridge[...,None,None]*torch.eye(u.shape[-1]).double()
            row['relative_recovery_difference']=float((h.double()-recovered).norm()/recovered.norm().clamp_min(1e-30))
            row['recovered_minimum_eigenvalue']=float(torch.linalg.eigvalsh(recovered).min())
        row['returned_minimum_eigenvalue']=float(torch.linalg.eigvalsh(h.double()).min())
        raw_records.append(row)
        return h
    m._joint_raw_metric=raw_observed
    def observed(h):
        u=original(h)
        row=dict(context,shape=list(h.shape),returned_none=u is None,caller=inspect.currentframe().f_back.f_code.co_name)
        damp=h.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-30)*m.GPTQ_DAMP
        hd=h+torch.eye(h.shape[-1],dtype=h.dtype)*damp[...,None,None]
        if u is None:
            row['minimum_damped_eigenvalue']=float(torch.linalg.eigvalsh(hd.double()).min())
            lower,info=torch.linalg.cholesky_ex(hd.double())
            row['double_cholesky_info']=info.tolist()
            if bool((info==0).all()):
                inverse_lower=torch.linalg.solve_triangular(lower,torch.eye(h.shape[-1]).double().expand_as(lower),upper=False)
                upper=torch.linalg.qr(inverse_lower,mode='r').R
                resid=hd.double()@(upper.transpose(-1,-2)@upper)-torch.eye(h.shape[-1]).double()
                row['alternative_factor_residual']=float(resid.abs().max())
        if u is not None:
            # Use the exact FP32 damped input of the live function, then audit in double.
            inv=u.double().transpose(-1,-2)@u.double()
            residual=hd.double()@inv-torch.eye(h.shape[-1],dtype=torch.float64)
            row.update(max_identity_residual=float(residual.abs().max()),
                       rms_identity_residual=float(residual.square().mean().sqrt()),
                       factor_finite=bool(torch.isfinite(u).all()))
        records.append(row)
        return u
    m._upper_cholesky_inv=observed
    for dataset,indices,isolated in (
        ('docs/papers/hif4-public-references/test/datasets/qwen3_5_35b_a3b_v1',[0] if args.failures_only else list(range(10)),False),
        ('local_fitted_models_attention_feedback_v2',[49] if args.failures_only else [0,24,49],True)):
        bank=torch.load(ROOT/dataset/'attn.pt',weights_only=True)
        for gi in indices:
            g=bank[gi];context.update(dataset=dataset,group=gi,isolated=isolated)
            role_calibration(m,g,isolated,'block_resample')
            print({'group':gi,'isolated':isolated,'observations':len(records)},flush=True)
    report={'completed':True,'parent':435,'observations':records,'raw_metric_observations':raw_records,
            'none':sum(r['returned_none'] for r in records),
            'max_residual':max((r.get('max_identity_residual',0) for r in records),default=0),
            'over_001':sum(r.get('max_identity_residual',0)>.01 for r in records),
            'warning':'Instrumented read-only factor observation on this host. Residual threshold is diagnostic, not a proof of online numerical behavior. Returned factors and calibration outputs were not replaced.'}
    filename='live-gptq-factor-metric-origin.json' if args.failures_only else 'live-gptq-factor-audit.json'
    (ROOT/'docs'/filename).write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k!='observations'})
if __name__=='__main__':main()
