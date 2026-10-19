"""Calibrated reciprocal pre-rotation scales with true GPTQ forward."""
import torch


def inverse_factor(h,damp):
    h=(h.double()+h.double().transpose(1,2))*.5
    eye=torch.eye(h.shape[-1],dtype=h.dtype).expand_as(h)
    ridge=h.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-30)*damp
    lower=torch.linalg.cholesky(h+ridge[:,None,None]*eye)
    b=torch.linalg.solve_triangular(lower,eye,upper=False)
    inv=b.transpose(1,2)@b
    return torch.linalg.cholesky((inv+inv.transpose(1,2))*.5,upper=True).float().contiguous()


def prepare(m,samples,parent,qh,kh,d,queries=32):
    hq=torch.zeros(kh,d,d);hk=torch.zeros_like(hq);data=[];eligible=[]
    for sample in samples[:-1]:
        rows=sample['q'][0].shape[0]
        if not 1<=rows<=1024:continue
        q=m.dequantize_nvfp4(*sample['q']).float().reshape(rows,qh,d).transpose(0,1)
        k=m.dequantize_nvfp4(*sample['k']).float().reshape(rows,kh,d).transpose(0,1)
        hq+=(q.transpose(1,2)@q).reshape(kh,qh//kh,d,d).sum(1)
        hk+=k.transpose(1,2)@k
        if rows>=2:eligible.append(sample)
    if not eligible or not bool((hq.diagonal(dim1=-2,dim2=-1).sum(-1)>0).all() and (hk.diagonal(dim1=-2,dim2=-1).sum(-1)>0).all()):
        return None
    for sample in eligible[-2:]:
        rows=sample['q'][0].shape[0];ix=torch.linspace(0,rows-1,min(rows,queries)).long()
        raw=[m.dequantize_nvfp4(*sample[r]).float() for r in 'qkv']
        value=m._deq_params(m.hif4_dynamic_quantize_v(*sample['v'],kh,d,parent['v_state']))
        target=m._joint_output(*raw,qh,kh,d,ix).reshape(-1,qh,d).transpose(0,1)
        data.append(dict(sample=sample,ix=ix,q=raw[0][ix].reshape(-1,qh,d).transpose(0,1),
                         k=raw[1].reshape(-1,kh,d).transpose(0,1),target=target,
                         value=value.reshape(rows,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)))
    return dict(hq=hq,hk=hk,data=data)


def bases(m,parent,kh,d,plain=False):
    result={}
    for role in 'qk':
        old=parent[role+'_state']
        if not plain and 'joint_transform' in old:
            base=old['joint_transform'].float()
        else:
            rotated=plain or old.get('rot')
            base=m._make_R(d,0 if plain else int(old.get('rseed') or 0)) if rotated else None
            if base is None:base=torch.eye(d)
            base=base[None].expand(kh,d,d)
        result[role]=base.contiguous()
    return result


def make_state(m,parent,context,basis,log_scale):
    scale=log_scale.detach().exp().contiguous();reciprocal=scale.reciprocal().contiguous()
    out=dict(parent)
    # Q's metric is K's transformed Gram, and conversely.
    for role,other,diagonal,field,applied in (('q','k',reciprocal,'qs',scale),('k','q',scale,'ks',reciprocal)):
        raw=context['hk' if role=='q' else 'hq']
        h=diagonal[:,:,None]*raw*diagonal[:,None,:]
        b=basis[other];h=b.transpose(1,2)@h@b;h=(h+h.transpose(1,2))*.5
        st=dict(parent[role+'_state'],joint_transform=basis[role].contiguous(),gq=1,
                u=inverse_factor(h,m.GPTQ_DAMP),H=h.bfloat16().contiguous())
        st[field]=applied
        out[role+'_state']=st
    return out


def forward(m,item,state,qh,kh,d):
    q=m._gate_query_values(item['sample'],state['q_state'],qh,kh,d,item['ix'])
    k=m._deq_params(m._gate_parent_k(*item['sample']['k'],kh,d,state['k_state']))
    return q.reshape(-1,qh,d).transpose(0,1),k.reshape(-1,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)


def loss(q,k,item,qh,kh,d):
    out=torch.softmax(q@k.transpose(1,2)/d**.5,-1)@item['value']
    return (out-item['target']).square().mean((1,2)).reshape(kh,qh//kh).mean(1)


def evaluate(m,context,state,qh,kh,d):
    return torch.stack([loss(*forward(m,item,state,qh,kh,d),item,qh,kh,d) for item in context['data']])


def train(m,samples,parent,qh,kh,d,plain=False,steps=4,rate=.05,queries=32):
    with torch.inference_mode(False),torch.enable_grad():
        with torch.no_grad():
            context=prepare(m,samples,parent,qh,kh,d,queries)
            if context is None:return None,None,dict(ineligible=True)
            basis=bases(m,parent,kh,d,plain)
            initial_log=parent['q_state']['qs'].float().log() if not plain and 'qs' in parent['q_state'] else torch.zeros(kh,d)
            initial_log=(initial_log-initial_log.mean(-1,keepdim=True)).clamp(-2.3,2.3)
            logs=initial_log.clone()
            initial=make_state(m,parent,context,basis,logs)
            initial_loss=evaluate(m,context,initial,qh,kh,d)
            norm=initial_loss.clamp_min(1e-20);best=(initial_loss/norm).mean(0)
        history=[best.tolist()];accepted=torch.zeros(kh,dtype=torch.int64)
        for _ in range(steps):
            logs=logs.detach().requires_grad_(True)
            with torch.no_grad():current=make_state(m,parent,context,basis,logs)
            objectives=[]
            for i,item in enumerate(context['data']):
                with torch.no_grad():qq,kk=forward(m,item,current,qh,kh,d)
                s=logs.exp()
                q=(item['q']*s.repeat_interleave(qh//kh,0)[:,None,:])@basis['q'].repeat_interleave(qh//kh,0)
                k=((item['k']/s[:,None,:])@basis['k']).repeat_interleave(qh//kh,0)
                q=qq+(q-q.detach());k=kk+(k-k.detach())
                objectives.append(loss(q,k,item,qh,kh,d)/norm[i])
            gradient=torch.autograd.grad(torch.stack(objectives).mean(),logs)[0]
            with torch.no_grad():
                gradient=gradient-gradient.mean(-1,keepdim=True)
                gradient=gradient/gradient.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-20)
                chosen=logs.detach();improved=False
                for fraction in (1.,.25):
                    trial=logs.detach()-rate*fraction*gradient
                    trial=(trial-trial.mean(-1,keepdim=True)).clamp(-2.3,2.3)
                    candidate=make_state(m,parent,context,basis,trial)
                    scores=(evaluate(m,context,candidate,qh,kh,d)/norm).mean(0)
                    take=torch.isfinite(scores)&(scores<best*.9999)
                    chosen=torch.where(take[:,None],trial,chosen);best=torch.where(take,scores,best)
                    accepted+=take.to(torch.int64);improved=improved or bool(take.any())
                logs=chosen.detach();history.append(best.tolist())
                if not improved:break
        with torch.no_grad():final=make_state(m,parent,context,basis,logs)
        return initial,final,dict(plain=plain,history=history,accepted=accepted.tolist(),
                                  training_rows=[x['sample']['q'][0].shape[0] for x in context['data']],
                                  max_log_change=float((logs-initial_log).abs().max()),
                                  reciprocal_error=float((final['q_state']['qs']*final['k_state']['ks']-1).abs().max()))
