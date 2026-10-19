"""Build independent post-calibration Attention candidates from pinned v422."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=(ROOT/'solution/solution_v422.py').read_text()
q=(ROOT/'solution/solution_v426.py').read_text()
q=q.replace('ATTN_REFINE_K_SWEEPS, quotient=True)','ATTN_REFINE_K_SWEEPS, quotient=False)')
q=q.replace('quotient=(role == "k")','quotient=(role == "k" and state.get("a429_deploy", False))')
q=q.replace("quotient=(role=='k')","quotient=(role=='k' and state.get('a429_deploy',False))")
q+='''

_a429_parent_calibration=hif4_calibration_attention

def _a429_mark(state):
    out=dict(state,a429_deploy=True)
    if isinstance(state.get('ts_fallback'),dict):
        out['ts_fallback']=_a429_mark(state['ts_fallback'])
    return out

@torch.inference_mode()
def hif4_calibration_attention(samples,qh,kh,d):
    parent=_a429_parent_calibration(samples,qh,kh,d)
    return dict(parent,k_state=_a429_mark(parent['k_state']))
'''
common='''

# Calibration is evaluated on the unchanged parent; only returned V state differs.
_ac_parent_calibration=hif4_calibration_attention
_ac_parent_v=hif4_dynamic_quantize_v
_ac_parent_probability=_vk_probabilities


def _ac_guard(hold,parent,trial_state,qh,kh,d):
    n=hold['v'][0].shape[0]
    ix=torch.linspace(0,n-1,min(32,n)).long()
    rawq=dequantize_nvfp4(hold['q'][0][ix],hold['q'][1][ix]).float()
    rawk=dequantize_nvfp4(*hold['k']).float()
    rawv=dequantize_nvfp4(*hold['v']).float()
    target=_joint_output(rawq,rawk,rawv,qh,kh,d,ix,query_subset=True)
    query=_joint_quant_queries(hold,parent['q_state'],qh,kh,d,ix)
    key=_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,parent['k_state']))
    losses=[]
    for st in (parent['v_state'],trial_state):
        v=_deq_params(hif4_dynamic_quantize_v(*hold['v'],kh,d,st))
        error=(_joint_output(query,key,v,qh,kh,d,ix,query_subset=True)-target).square().reshape(len(ix),kh,-1).mean(-1)
        losses.append(torch.stack((error[::2].mean(0),error[1::2].mean(0))))
    return (torch.isfinite(losses[1]).all(0)&(losses[1]<losses[0]*.99).all(0))


def _ac_eligible(samples,parent,qh,kh,d):
    st=parent['v_state']
    return (len(samples)>=3 and isinstance(st,dict) and 'kernel' in st
        and isinstance(st.get('active'),torch.Tensor) and bool(st['active'].any())
        and 4<=samples[-1]['v'][0].shape[0]<=2048
        and samples[-1]['v'][0].shape[0]*qh*d<=2**23)
'''
temp='''

def _a430_power(probability,beta):
    logits=probability.clamp_min(1.e-30).log()*beta[:,None,None]
    return torch.softmax(logits,-1)


def _a430_fit(samples,state,qh,kh,d):
    estimates=[]
    for sample in samples[:-1][-3:]:
        n=sample['v'][0].shape[0]
        if not 4<=n<=2048 or n*qh*d>2**23:continue
        ix=torch.linspace(0,n-1,min(16,n)).long()
        q=dequantize_nvfp4(sample['q'][0][ix],sample['q'][1][ix]).float().reshape(-1,kh,qh//kh,d).permute(1,2,0,3)
        k=dequantize_nvfp4(*sample['k']).float().reshape(n,kh,d).transpose(0,1)
        p=torch.softmax(q @ k[:,None].transpose(-1,-2)/d**.5,-1)
        concentration=p.square().sum(-1).mean((1,2))
        v=dequantize_nvfp4(*sample['v']).float()
        proxy=_ac_parent_probability(v,state,kh)
        lo=torch.full((kh,),.25);hi=torch.full((kh,),4.)
        for _ in range(10):
            mid=(lo+hi)*.5
            observed=_a430_power(proxy,mid).square().sum(-1).mean(-1)
            low=observed<concentration
            lo=torch.where(low,mid,lo);hi=torch.where(low,hi,mid)
        estimates.append(((lo+hi)*.5).log())
    if len(estimates)<2:return None
    # Cross-sequence central estimate; no absolute token position is stored.
    return torch.stack(estimates).median(0).values.exp().float().contiguous()


def _vk_probabilities(x,state,kh):
    p=_ac_parent_probability(x,state,kh)
    beta=state.get('a430_temperature')
    if not isinstance(beta,torch.Tensor):return p
    adjusted=_a430_power(p,beta.float())
    mask=state['a430_active'][:,None,None]
    return torch.where(mask,adjusted,p)


@torch.inference_mode()
def hif4_calibration_attention(samples,qh,kh,d):
    parent=_ac_parent_calibration(samples,qh,kh,d)
    if not _ac_eligible(samples,parent,qh,kh,d):return parent
    try:
        st=parent['v_state'];beta=_a430_fit(samples,st,qh,kh,d)
        if beta is None or not bool(torch.isfinite(beta).all()):return parent
        trial=dict(st,a430_temperature=beta,a430_active=st['active'].clone())
        active=_ac_guard(samples[-1],parent,trial,qh,kh,d)&st['active']
        if bool(active.any()):return dict(parent,v_state=dict(trial,a430_active=active.contiguous()))
    except (RuntimeError,ValueError,KeyError,TypeError):
        pass
    return parent
'''
tree='''

def _a431_tree(x,params,state,kh,d):
    t=x.shape[0]
    if not 2<=t<=4096 or t*kh*d>2**23 or d%64:return params
    probability=_ac_parent_probability(x,state,kh)
    if probability.shape[1]>128 or not bool(torch.isfinite(probability).all()):return params
    active=state['a431_active']
    values=_deq_params(params).float().reshape(t,kh,d).transpose(0,1).contiguous()
    target=x.reshape(t,kh,d).transpose(0,1).contiguous()
    step=(_params_unit_flat(params)*.25).reshape(t,kh,d).transpose(0,1)
    old_lv3=params['scale_lv3'].reshape(t,kh,d//4).transpose(0,1)
    new_lv3=3-old_lv3
    new_step=step*(new_lv3/old_lv3).repeat_interleave(4,-1)
    error=values-target
    gradient=probability.transpose(1,2)@(probability@error)+error/t
    diagonal=probability.square().sum(1)[:,:,None]+1./t
    optimum=values-gradient/diagonal.clamp_min(1.e-30)
    codes=(optimum/new_step.clamp_min(1.e-30)).round().clamp(-7,7)
    candidate=codes*new_step
    delta=candidate-values
    change=2*gradient*delta+diagonal*delta.square()
    cap=error.square().mean(1).clamp_min(1.e-30)
    new_mse=cap[:,None]+(2*error*delta+delta.square())/t
    valid=(new_mse<=cap[:,None]*1.05).reshape(kh,t,-1,4).all(-1)
    score=change.reshape(kh,t,-1,4).sum(-1)
    score=score.masked_fill(~valid,torch.inf)
    best,token=score.min(1)
    take=(best<0)&torch.isfinite(best)&active[:,None]
    if not bool(take.any()):return params
    # Distinct four-channel groups never share output features: simultaneous
    # one-token updates per group have no omitted token cross term.
    hh=torch.arange(kh)[:,None];gg=torch.arange(d//4)[None]
    result=values.reshape(kh,t,d//4,4).clone()
    proposed=candidate.reshape(kh,t,d//4,4)
    result[hh,token,gg]=torch.where(take[:,:,None],proposed[hh,token,gg],result[hh,token,gg])
    lv3=old_lv3.clone()
    lv3[hh,token,gg]=torch.where(take,new_lv3[hh,token,gg],lv3[hh,token,gg])
    result=result.reshape(kh,t,d)
    # Decode-grid parameters are reconstructed directly, without old-step recoding.
    steps=step*(lv3/old_lv3).repeat_interleave(4,-1)
    final_codes=(result/steps.clamp_min(1.e-30)).round().clamp(-7,7).transpose(0,1).contiguous()
    return dict(params,scale_lv3=lv3.transpose(0,1).reshape_as(params['scale_lv3']).to(params['scale_lv3'].dtype).contiguous(),
                sign=final_codes.sign().reshape_as(params['sign']).to(params['sign'].dtype).contiguous(),
                mant=(final_codes.abs()*.25).reshape_as(params['mant']).to(params['mant'].dtype).contiguous())


def hif4_dynamic_quantize_v(vq,vs,kh,d,state):
    params=_ac_parent_v(vq,vs,kh,d,state)
    if not isinstance(state,dict) or 'a431_active' not in state:return params
    return _a431_tree(dequantize_nvfp4(vq,vs).float(),params,state,kh,d)


@torch.inference_mode()
def hif4_calibration_attention(samples,qh,kh,d):
    parent=_ac_parent_calibration(samples,qh,kh,d)
    if not _ac_eligible(samples,parent,qh,kh,d):return parent
    try:
        st=parent['v_state'];trial=dict(st,a431_active=st['active'].clone())
        active=_ac_guard(samples[-1],parent,trial,qh,kh,d)&st['active']
        if bool(active.any()):return dict(parent,v_state=dict(trial,a431_active=active.contiguous()))
    except (RuntimeError,ValueError,KeyError,TypeError):
        pass
    return parent
'''
for version,source in ((429,q),(430,base+common+temp),(431,base+common+tree)):
    p=ROOT/f'solution/solution_v{version}.py'
    if p.exists():raise RuntimeError(f'Existing {p}, do not overwrite')
    ast.parse(source);p.write_text(source)
