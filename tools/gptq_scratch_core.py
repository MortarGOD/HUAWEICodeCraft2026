"""Column-major signed-code scratch; same rounding and error update order."""


def _gptq_quantize_batched_np(x, unit, hinv):
    n=x.shape[-1]; per_batch=hinv.dim()==3
    W=x.clone(); Q=torch.empty_like(W)
    unp=unit.contiguous().numpy(); hnp=hinv.contiguous().numpy()
    four=np.float32(4.0); quarter=np.float32(.25)
    for i1 in range(0,n,GPTQ_BLOCK):
        i2=min(i1+GPTQ_BLOCK,n)
        W1=W[...,i1:i2].permute(2,0,1).contiguous()
        Q1=torch.empty_like(W1); E1=torch.empty_like(W1)
        w1,q1,e1=W1.numpy(),Q1.numpy(),E1.numpy()
        Hi=hnp[...,i1:i2,i1:i2];u=unp[...,i1:i2]
        for i in range(i2-i1):
            w,ui,q=w1[i],u[...,i],q1[i]
            # Symmetric round-to-even: signed clipping is identical to
            # abs/round/clip/sign, without allocating those intermediates.
            np.divide(w,ui,out=q)
            np.multiply(q,four,out=q)
            np.rint(q,out=q)
            np.clip(q,-7.0,7.0,out=q)
            np.multiply(q,quarter,out=q)
            np.multiply(q,ui,out=q)
            diag=np.maximum(Hi[...,i,i],np.float32(1e-30))
            if per_batch:diag=diag[...,None]
            np.subtract(w,q,out=e1[i])
            np.divide(e1[i],diag,out=e1[i])
            if i<i2-i1-1:
                seg=Hi[...,i,i+1:]
                seg=seg.T[:,:,None] if per_batch else seg[:,None,None]
                w1[i+1:]-=e1[i][None,...]*seg
        Q[...,i1:i2]=Q1.permute(1,2,0)
        if i2<n:
            W[...,i2:]-=torch.matmul(E1.permute(1,2,0).contiguous(),hinv[...,i1:i2,i2:])
    return Q


def _gptq_quantize_batched_columns(x, unit, hinv):
    n=x.shape[-1];per_batch=hinv.dim()==3
    W=x.clone();Q=torch.empty_like(W)
    for i1 in range(0,n,GPTQ_BLOCK):
        i2=min(i1+GPTQ_BLOCK,n)
        W1=W[...,i1:i2].permute(2,0,1).contiguous()
        Q1=torch.empty_like(W1);E1=torch.empty_like(W1)
        Hi=hinv[...,i1:i2,i1:i2]
        u=unit[...,i1:i2].permute(2,0,1).contiguous()
        for i in range(i2-i1):
            w,ui,q=W1[i],u[i],Q1[i]
            torch.div(w,ui,out=q)
            q.mul_(4.0).round_().clamp_(-7.0,7.0).mul_(.25).mul_(ui)
            diag=Hi[...,i,i].clamp_min(1e-30)
            if per_batch:diag=diag.unsqueeze(-1)
            torch.sub(w,q,out=E1[i]);E1[i].div_(diag)
            if i<i2-i1-1:
                seg=Hi[...,i,i+1:]
                seg=seg.T.unsqueeze(-1) if per_batch else seg[:,None,None]
                W1[i+1:]-=E1[i].unsqueeze(0)*seg
        Q[...,i1:i2]=Q1.permute(1,2,0)
        if i2<n:
            W[...,i2:]-=torch.matmul(E1.permute(1,2,0).contiguous(),hinv[...,i1:i2,i2:])
    return Q
