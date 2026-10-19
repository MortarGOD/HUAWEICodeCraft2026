"""Fixed-lattice K updates to compensate final K/V output error.

Queries may be a calibration distribution or an explicitly labeled offline
oracle. This kernel itself does not read other interfaces or fit test labels.
"""
import torch


def output_gradient(queries,keys,values,target):
    dim=keys.shape[-1]
    logits=queries@keys.transpose(-1,-2)/dim**.5
    p=torch.softmax(logits,-1)
    output=p@values
    error=output-target
    sensitivity=error@values.transpose(-1,-2)
    dz=p*(sensitivity-(p*sensitivity).sum(-1,keepdim=True))
    gradient=dz.transpose(-1,-2)@queries/dim**.5
    distance=(values.square().sum(-1)[:,None,:]-2*(output@values.transpose(-1,-2))
              +output.square().sum(-1)[:,:,None]).clamp_min(0)
    diagonal=(p.square()*distance).transpose(-1,-2)@queries.square()/dim
    return gradient,diagonal,output


def refine_keys(queries,raw_keys,quant_keys,raw_values,quant_values,step,rounds=8,logit_cap=2.):
    dim=raw_keys.shape[-1]
    reference_logits=queries@raw_keys.transpose(-1,-2)/dim**.5
    target=torch.softmax(reference_logits,-1)@raw_values
    codes=(quant_keys/step).round().clamp(-7,7)
    initial_logits=queries@(codes*step).transpose(-1,-2)/dim**.5
    logit_limit=(initial_logits-reference_logits).square().mean((-1,-2)).clamp_min(1e-12)*logit_cap
    initial=(torch.softmax(initial_logits,-1)@quant_values-target).square().mean((-1,-2))
    best=initial
    history=[best.clone()]
    rows=raw_keys.shape[-2]
    budgets=list(dict.fromkeys((rows,min(rows,32),min(rows,8))))
    accepted=torch.zeros(queries.shape[0],dtype=torch.int64)
    for _ in range(rounds):
        gradient,diagonal,_=output_gradient(queries,codes*step,quant_values,target)
        direction=torch.where(gradient>=0,-torch.ones_like(gradient),torch.ones_like(gradient))
        illegal=((direction>0)&(codes>=7))|((direction<0)&(codes<=-7))
        cost=2*gradient*direction*step+diagonal*step.square()
        cost=cost.masked_fill(illegal,float('inf'))
        gain,index=cost.min(-1)
        movement=direction.gather(-1,index[:,:,None]).squeeze(-1)
        eligible=gain<0
        next_codes=codes
        for budget in budgets:
            mask=eligible
            if budget<rows:
                selected=gain.topk(budget,dim=-1,largest=False).indices
                mask=torch.zeros_like(eligible).scatter_(1,selected,True)&eligible
            candidate=codes.clone().scatter_add_(-1,index[:,:,None],(movement*mask)[:,:,None])
            logits=queries@(candidate*step).transpose(-1,-2)/dim**.5
            loss=(torch.softmax(logits,-1)@quant_values-target).square().mean((-1,-2))
            distortion=(logits-reference_logits).square().mean((-1,-2))
            take=(loss<best*.999999)&(distortion<=logit_limit)
            next_codes=torch.where(take[:,None,None],candidate,next_codes)
            accepted+=take.to(torch.int64)
            best=torch.where(take,loss,best)
        codes=next_codes
        history.append(best.clone())
    return codes*step,dict(initial=initial,final=best,accepted=accepted,history=torch.stack(history),
                          logit_limit=logit_limit)
