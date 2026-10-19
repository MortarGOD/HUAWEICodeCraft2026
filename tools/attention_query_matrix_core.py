"""Full Q channel matrix trained against actual quantized attention output."""
QM_DEBUG={}
_qm_parent_calibration=hif4_calibration_attention


def _qm_state(st,matrix,d,kh):
    if 'joint_transform' in st:base=st['joint_transform'].float()
    elif st.get('rot'):base=_make_R(d,int(st.get('rseed') or 0)).expand(kh,-1,-1)
    else:base=torch.eye(d).expand(kh,-1,-1)
    return dict(st,joint_transform=(base@matrix).contiguous())


def _qm_loss_gradient(q,k,v,target):
    # Per query-head MSE; derivative at the actual quantized forward value.
    d=q.shape[-1];p=torch.softmax(q@k.transpose(1,2)/d**.5,-1)
    error=p@v-target
    loss=error.square().mean((1,2))
    gp=(2*error/(q.shape[1]*d))@v.transpose(1,2)
    gz=p*(gp-(gp*p).sum(-1,keepdim=True))
    return loss,(gz@k)/d**.5,error.square().mean(-1)


def _qm_data(sample,states,qh,kh,d,count):
    n=len(sample['q'][0]);ix=torch.linspace(0,n-1,min(n,count)).long()
    raw=[dequantize_nvfp4(sample['q'][0][ix],sample['q'][1][ix]).float()]
    raw += [dequantize_nvfp4(*sample[r]).float() for r in 'kv']
    target=_joint_output(*raw,qh,kh,d,ix,query_subset=True).reshape(len(ix),qh,d).transpose(0,1)
    x=_kt_input(raw[0],states['q_state'],qh,d,'q').reshape(len(ix),qh,d).transpose(0,1)
    kv=[_deq_params(fn(*sample[r],kh,d,states[r+'_state'])).reshape(n,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
        for r,fn in [('k',hif4_dynamic_quantize_k),('v',hif4_dynamic_quantize_v)]]
    return dict(sample=sample,ix=ix,x=x,k=kv[0],v=kv[1],target=target)


def _qm_eval(data,st,matrix,qh,kh,d):
    state=_qm_state(st,matrix,d,kh);rows=[];grad=[];folds=[]
    for item in data:
        q=_joint_quant_queries(item['sample'],state,qh,kh,d,item['ix']).reshape(-1,qh,d).transpose(0,1)
        loss,g,e=_qm_loss_gradient(q,item['k'],item['v'],item['target'])
        rows.append(loss.reshape(kh,qh//kh).mean(-1))
        grad.append((item['x'].transpose(1,2)@g).reshape(kh,qh//kh,d,d).mean(1))
        e=e.reshape(kh,qh//kh,-1).mean(1)
        folds.append(torch.stack((e[:,::2].mean(-1),e[:,1::2].mean(-1))))
    return torch.stack(rows),torch.stack(grad),torch.stack(folds)


def _qm_tangent(q,k,v,dq,target):
    p=torch.softmax(q@k.transpose(1,2)/q.shape[-1]**.5,-1)
    dz=dq@k.transpose(1,2)/q.shape[-1]**.5
    tangent=(p*(dz-(p*dz).sum(-1,keepdim=True)))@v
    return -(tangent*(p@v-target)).mean((1,2)),tangent.square().mean((1,2))


def _qm_step(data,st,matrix,direction,norm,qh,kh,d):
    numerator=torch.zeros(kh);denominator=torch.zeros(kh)
    state=_qm_state(st,matrix,d,kh)
    for i,item in enumerate(data):
        q=_joint_quant_queries(item['sample'],state,qh,kh,d,item['ix']).reshape(-1,qh,d).transpose(0,1)
        dq=item['x']@direction.repeat_interleave(qh//kh,0)
        a,b=_qm_tangent(q,item['k'],item['v'],dq,item['target'])
        numerator+=a.reshape(kh,qh//kh).mean(-1)/norm[i]
        denominator+=b.reshape(kh,qh//kh).mean(-1)/norm[i]
    return (numerator/denominator.clamp_min(1.e-20)).clamp(0,.1*d**.5)


@torch.inference_mode()
def hif4_calibration_attention(samples,q_num_heads,kv_num_heads,head_dim):
    parent=_qm_parent_calibration(samples,q_num_heads,kv_num_heads,head_dim)
    qh,kh,d=q_num_heads,kv_num_heads,head_dim;st=parent['q_state']
    QM_DEBUG.clear();QM_DEBUG.update(accepted=0,error=False)
    if len(samples)<3 or not st.get('gq') or 'u' not in st or d%64:return parent
    selected=samples[:-1][-2:]
    if any(not 4<=len(s['q'][0])<=4096 for s in selected+[samples[-1]]):return parent
    try:
        data=[_qm_data(s,parent,qh,kh,d,16) for s in selected]
        eye=torch.eye(d).expand(kh,-1,-1).clone();matrix=eye.clone()
        initial,gradient,_=_qm_eval(data,st,matrix,qh,kh,d)
        norm=initial.clamp_min(1.e-20);best=torch.ones(kh);history=[best.tolist()]
        for _ in range(4):
            direction=(gradient/norm[:,:,None,None]).mean(0)
            direction=-direction/direction.square().sum((-1,-2)).sqrt()[:,None,None].clamp_min(1.e-20)
            step=_qm_step(data,st,matrix,direction,norm,qh,kh,d)
            chosen=matrix;chosen_gradient=gradient;any_take=False
            for fraction in (1.,.25):
                proposal=matrix+fraction*step[:,None,None]*direction
                delta=proposal-eye
                # Bound the total Frobenius displacement to 10% of identity.
                factor=(.1*d**.5/delta.square().sum((-1,-2)).sqrt().clamp_min(1.e-20)).clamp_max(1)
                proposal=eye+delta*factor[:,None,None]
                loss,g,_=_qm_eval(data,st,proposal,qh,kh,d)
                risk=(loss/norm).mean(0)
                take=torch.isfinite(risk)&(risk<best*.9999)
                chosen=torch.where(take[:,None,None],proposal,chosen)
                chosen_gradient=torch.where(take[None,:,None,None],g,chosen_gradient)
                best=torch.where(take,risk,best);any_take=any_take or bool(take.any())
            matrix=chosen;gradient=chosen_gradient;history.append(best.tolist())
            if not any_take:break
        hold=[_qm_data(samples[-1],parent,qh,kh,d,32)]
        _,_,before=_qm_eval(hold,st,eye,qh,kh,d)
        _,_,after=_qm_eval(hold,st,matrix,qh,kh,d)
        take=torch.isfinite(after[0]).all(0)&torch.isfinite(before[0]).all(0)&(after[0]<before[0]*.995).all(0)
        QM_DEBUG.update(accepted=int(take.sum()),history=history,hold_ratios=(after[0].mean(0)/before[0].mean(0).clamp_min(1.e-30)).tolist())
        if bool(take.any()):
            final=torch.where(take[:,None,None],matrix,eye)
            return dict(parent,q_state=_qm_state(st,final,d,kh))
    except (RuntimeError,ValueError) as exc:
        QM_DEBUG.update(error=True,error_type=type(exc).__name__)
    return parent
