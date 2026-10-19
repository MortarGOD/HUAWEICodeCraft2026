"""A4 coordinate compatibility and A5 frozen historical sign audit; no fitting."""
import copy,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.check_role_decoy_calibration import role_samples
from tools.verify_v426 import equal
from hif4_quant import dequantize_nvfp4,dequantize_hif4,quantize_to_hif4
from evaluate_local import gqa_attention,mse
from self_check import validate_attention_calibration_result,validate_hif4_params


def calibrate(version,group,mode):
    qh,kh,d=[group[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
    m=load(version,'audit'+mode)
    if mode=='paired':return m,m.hif4_calibration_attention(group['calib'],qh,kh,d)
    states={}
    for role in 'qkv':
        m=load(version,'audit'+mode+role)
        result=m.hif4_calibration_attention(role_samples(group['calib'],role,mode,829+ord(role)),qh,kh,d)
        states[role+'_state']=result[role+'_state']
    return m,states


def quality(m,states,sample,qh,kh,d):
    raw={r:dequantize_nvfp4(*sample[r]).float() for r in 'qkv'}
    base={r:dequantize_hif4(quantize_to_hif4(raw[r]),raw[r].shape) for r in 'qkv'}
    target=gqa_attention(*[raw[r] for r in 'qkv'],qh,kh,d)
    denominator=mse(target,gqa_attention(*[base[r] for r in 'qkv'],qh,kh,d))
    values={}
    for r in 'vkq':
        p=getattr(m,'hif4_dynamic_quantize_'+r)(*sample[r],qh if r=='q' else kh,d,states[r+'_state'])
        assert not validate_hif4_params(p,raw[r].shape,'audit')
        values[r]=dequantize_hif4(p,raw[r].shape)
    cost=mse(target,gqa_attention(*[values[r] for r in 'qkv'],qh,kh,d))
    return 1-cost/denominator if denominator else 0.


def main():
    torch.set_num_threads(4)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    output=ROOT/'docs/attention-proxy-audit-20260908.json'
    report={'completed':False,'selected_groups':[0,24,49],'historical':[],'coordinates':[],
      'warning':'Fixed systematic source indices, not selected by outcomes. Three decoy assumptions are not the online generator. Seven labels cannot validate a fitted forecast.'}
    versions=(411,412,413,415,416,417,418,419)
    # Paired sign audit over 15 tests; two decoy modes over five tests each.
    for mode,indices in (('paired',(0,24,49)),('row_permute',(24,)),('block_resample',(24,))):
        for gi in indices:
            g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
            for v in versions:
                started=time.perf_counter();m,states=calibrate(v,g,mode)
                assert not validate_attention_calibration_result(states,'history')
                scores=[quality(m,states,s,qh,kh,d) for s in g['test']]
                report['historical'].append({'mode':mode,'group':gi,'version':v,'quality':scores,'seconds':time.perf_counter()-started})
                output.write_text(json.dumps(report,indent=2)+'\n');print(mode,gi,v,sum(scores),flush=True)
    for mode in ('paired','row_permute','block_resample'):
        for gi in (0,24,49):
            g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
            m,st=calibrate(422,g,mode);maps=[]
            for role,nh in (('q',qh),('k',kh)):
                state=dict(st[role+'_state']);state.pop('temperature_gain',None)
                mat=m._kt_input(torch.eye(d).repeat(1,nh),state,nh,d,role).reshape(d,nh,d).transpose(0,1)
                maps.append(mat)
            product=maps[0]@maps[1].repeat_interleave(qh//kh,0).transpose(-1,-2)
            err=(product-torch.eye(d)).square().mean((-1,-2)).sqrt()*d**.5
            report['coordinates'].append({'mode':mode,'group':gi,'relative_identity_error':err.tolist(),'scope':'top-level pre-translation linear maps, temperature removed; not full deployed quantization'})
            output.write_text(json.dumps(report,indent=2)+'\n')
    report['sign_agreement']={}
    for mode in ('paired','row_permute','block_resample'):
        sums={v:sum(sum(r['quality']) for r in report['historical'] if r['mode']==mode and r['version']==v) for v in versions}
        pairs=[(412,411,1),(413,412,-1),(415,412,-1),(416,412,-1),(417,412,-1),(418,412,-1),(419,412,1)]
        values=[{'child':c,'parent':p,'online_sign':sign,'local_delta':sums[c]-sums[p],'agrees':(sums[c]-sums[p])*sign>1e-12} for c,p,sign in pairs]
        report['sign_agreement'][mode]={'comparisons':values,'correct':sum(r['agrees'] for r in values),'out_of':7}
    report['completed']=True;output.write_text(json.dumps(report,indent=2)+'\n');print(report['sign_agreement'],flush=True)
if __name__=='__main__':main()
