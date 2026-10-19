"""Factor-core override selecting queries by observed calibration output error."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def output_risk_source():
    source=(ROOT/'tools/attention_factor_risk_core.py').read_text()
    assert source.count('_AF_SEARCH_RIDGE=.1')==1
    source=source.replace('_AF_SEARCH_RIDGE=.1','_AF_SEARCH_RIDGE=1.',1)
    source=source.replace('    proto=[];xs=[];ks=[]','    proto=[];xs=[];ks=[];risks=[]',1)
    old="""        proto.append(q)
        xs.append(dequantize_nvfp4(*sample['v']).float().reshape(t,kh,d).transpose(0,1))
        ks.append(_deq_params(hif4_dynamic_quantize_k(*sample['k'],kh,d,parent['k_state']))
                  .reshape(t,kh,d).transpose(0,1))"""
    new="""        proto.append(q)
        x=dequantize_nvfp4(*sample['v']).float().reshape(t,kh,d).transpose(0,1)
        key=_deq_params(hif4_dynamic_quantize_k(*sample['k'],kh,d,parent['k_state'])).reshape(t,kh,d).transpose(0,1)
        base=_deq_params(_am_parent_v(*sample['v'],kh,d,parent['v_state'])).reshape(t,kh,d).transpose(0,1)
        probability=(q@key.transpose(1,2)/(d**.5)).softmax(-1)
        risks.append((probability@(base-x)).square().mean(-1))
        xs.append(x);ks.append(key)"""
    assert source.count(old)==1;source=source.replace(old,new,1)
    old="""    proto=torch.cat(proto,1)
    ix=torch.linspace(0,proto.shape[1]-1,min(_AF_RANK,proto.shape[1])).long()
    proto=proto[:,ix];x=torch.cat(xs,1);k=torch.cat(ks,1)"""
    new="""    proto=torch.cat(proto,1);risk=torch.cat(risks,1)
    rank=min(_AF_RANK,proto.shape[1]);ix=risk.topk(rank,1,largest=True,sorted=True).indices
    proto=proto.gather(1,ix[:,:,None].expand(-1,-1,d));x=torch.cat(xs,1);k=torch.cat(ks,1)"""
    assert source.count(old)==1;source=source.replace(old,new,1)
    return source


def output_risk_weight_source():
    source=(ROOT/'tools/attention_factor_risk_core.py').read_text()
    assert source.count('_AF_SEARCH_RIDGE=.1')==1
    source=source.replace('_AF_SEARCH_RIDGE=.1','_AF_SEARCH_RIDGE=1.',1)
    source=source.replace('    proto=[];xs=[];ks=[]','    proto=[];xs=[];ks=[];risks=[]',1)
    old="""        proto.append(q)
        xs.append(dequantize_nvfp4(*sample['v']).float().reshape(t,kh,d).transpose(0,1))
        ks.append(_deq_params(hif4_dynamic_quantize_k(*sample['k'],kh,d,parent['k_state']))
                  .reshape(t,kh,d).transpose(0,1))"""
    new="""        proto.append(q)
        x=dequantize_nvfp4(*sample['v']).float().reshape(t,kh,d).transpose(0,1)
        key=_deq_params(hif4_dynamic_quantize_k(*sample['k'],kh,d,parent['k_state'])).reshape(t,kh,d).transpose(0,1)
        base=_deq_params(_am_parent_v(*sample['v'],kh,d,parent['v_state'])).reshape(t,kh,d).transpose(0,1)
        probability=(q@key.transpose(1,2)/(d**.5)).softmax(-1)
        risks.append((probability@(base-x)).square().mean(-1))
        xs.append(x);ks.append(key)"""
    assert source.count(old)==1;source=source.replace(old,new,1)
    old="""    proto=torch.cat(proto,1)
    ix=torch.linspace(0,proto.shape[1]-1,min(_AF_RANK,proto.shape[1])).long()
    proto=proto[:,ix];x=torch.cat(xs,1);k=torch.cat(ks,1)"""
    new="""    proto=torch.cat(proto,1);risk=torch.cat(risks,1)
    ix=torch.linspace(0,proto.shape[1]-1,min(_AF_RANK,proto.shape[1])).long()
    proto=proto[:,ix];risk=risk.gather(1,ix[None].expand(kh,-1))
    query_weights=(risk/risk.mean(1,keepdim=True).clamp_min(1e-30)).sqrt().clamp(.5,2.)
    query_weights=query_weights/query_weights.sum(1,keepdim=True)
    x=torch.cat(xs,1);k=torch.cat(ks,1)"""
    assert source.count(old)==1;source=source.replace(old,new,1)
    source=source.replace('st.update(beta=beta.contiguous(),search_ridge=_AF_SEARCH_RIDGE)',
        'st.update(beta=beta.contiguous(),search_ridge=_AF_SEARCH_RIDGE,query_weights=query_weights.contiguous())',1)
    old="""def _af_probability(v,st,kh,d):
    x=v.reshape(-1,kh,d).transpose(0,1)
    logits=(_af_features(x,st)@st['beta']).transpose(1,2)
    return logits.clamp(-40.,40.).softmax(-1)


def _am_moments(v,st,kh,d):
    p=_af_probability(v,st,kh,d)
    return p.mean(1),p.var(1,unbiased=False),torch.zeros(kh,v.shape[0],d)"""
    new="""def _af_probability(v,st,kh,d):
    x=v.reshape(-1,kh,d).transpose(0,1)
    logits=(_af_features(x,st)@st['beta']).transpose(1,2)
    p=logits.clamp(-40.,40.).softmax(-1)
    return p*(st['query_weights']*p.shape[1]).sqrt()[:,:,None]


def _am_moments(v,st,kh,d):
    b=_af_probability(v,st,kh,d);w=st['query_weights']
    mean=(b*(w/b.shape[1]).sqrt()[:,:,None]).sum(1)
    variance=(b.square().mean(1)-mean.square()).clamp_min(0.)
    return mean,variance,torch.zeros(kh,v.shape[0],d)"""
    assert source.count(old)==1;return source.replace(old,new,1)
