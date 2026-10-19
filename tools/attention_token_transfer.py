"""A2 frozen token Hessian prototype and independent-decoy transfer diagnostic."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hif4_quant import dequantize_nvfp4
from tools.check_role_decoy_calibration import role_samples

def hessian(sample,qh,kh,d):
    q=dequantize_nvfp4(*sample['q']).float().reshape(-1,kh,qh//kh,d).permute(1,2,0,3)
    k=dequantize_nvfp4(*sample['k']).float().reshape(-1,kh,d).permute(1,0,2)
    p=torch.softmax(q@k[:,None].transpose(-1,-2)/d**.5,-1)
    return torch.einsum('hgqt,hgqu->htu',p,p)/(p.shape[1]*p.shape[2])

def similarity(a,b):
    n=a.shape[-1];mask=~torch.eye(n,dtype=torch.bool)
    aa=a[:,mask];bb=b[:,mask]
    aa=aa-aa.mean(-1,keepdim=True);bb=bb-bb.mean(-1,keepdim=True)
    denom=aa.norm(dim=-1)*bb.norm(dim=-1)
    value=(aa*bb).sum(-1)/denom.clamp_min(1e-30)
    return value[denom>1e-15].tolist()

def frozen_token_step(x,y,step,H):
    """One exact coordinate per feature, retained as offline A2 reference."""
    metric=.7*H/H.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-30)[:,None,None]+.3*torch.eye(H.shape[-1])
    e=y-x;gradient=metric@e;diag=metric.diagonal(dim1=-2,dim2=-1)[:,:,None]
    codes=((y-gradient/diag)/step).round().clamp(-7,7)
    delta=codes*step-y;change=2*gradient*delta+diag*delta.square()
    gain,ix=change.min(1);out=y.clone();hh=torch.arange(y.shape[0])[:,None];cc=torch.arange(y.shape[-1])[None]
    out[hh,ix,cc]=torch.where(gain<0,(codes*step)[hh,ix,cc],y[hh,ix,cc])
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--datasets',nargs='+',default=['local_fitted_models_attention_feedback_v2','local_fitted_models_covariance_v1'])
    parser.add_argument('--output',default='docs/attention-token-transfer-20260908.json')
    args=parser.parse_args()
    torch.set_num_threads(4);rows=[]
    for bank in args.datasets:
        groups=torch.load(ROOT/bank/'attn.pt',weights_only=True)
        for gi,g in enumerate(groups):
            qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
            for si,s in enumerate(g['calib'][:-1]):
                n=s['v'][0].shape[0]
                if not 4<=n<=128:continue
                h=hessian(s,qh,kh,d)
                for mode in ('row_permute','block_resample'):
                    fake=role_samples([s],'v',mode,829+gi*7+si)[0]
                    rows.append({'bank':bank,'group':gi,'tokens':n,'mode':mode,'cosine':similarity(h,hessian(fake,qh,kh,d))})
                for test in g['test']:
                    if test['v'][0].shape[0]==n:
                        rows.append({'bank':bank,'group':gi,'tokens':n,'mode':'new_sequence','cosine':similarity(h,hessian(test,qh,kh,d))})
    vals=[v for row in rows for v in row['cosine']]
    passed=bool(vals) and sum(v>=.5 for v in vals)/len(vals)>=.8
    report={'rows':rows,'preregistered_cosine_threshold':.5,'required_fraction':.8,'observed_fraction':sum(v>=.5 for v in vals)/max(1,len(vals)),'comparisons':len(vals),'eligible_for_frozen_position_H':passed,'warning':'Cosine gate is necessary, not sufficient. Decoy source unknown. No fitted online score.'}
    (ROOT/args.output).write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='rows'})
if __name__=='__main__':main()
