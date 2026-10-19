"""Calibrated fixed-decoder block reconstruction with an anchored prior."""

def _linear_decoder_cd(module, weight_params, z, y, rounds=1, ridge=.1):
    import torch
    params={name:value.clone() for name,value in weight_params.items()}
    b=module._linear_score_decode(params).float()
    anchor=b.clone();residual=y-z@b.T
    outputs,inputs=b.shape;blocks=inputs//64
    zb=z.reshape(-1,blocks,64)
    grams=torch.einsum('rbc,rbd->bcd',zb,zb)
    lam=ridge*grams.diagonal(dim1=-2,dim2=-1).mean().clamp_min(1e-20)
    regularized=grams+lam*torch.eye(64)[None]
    inverse=torch.linalg.inv(regularized)
    order=grams.diagonal(dim1=-2,dim2=-1).sum(-1).argsort(descending=True).tolist()
    for _ in range(rounds):
        for block in order:
            begin=block*64;end=begin+64;old=b[:,begin:end];x=zb[:,block]
            gradient=residual.T@x-lam*(old-anchor[:,begin:end])
            proposal=old+gradient@inverse[block]
            candidate=module._quantize_linear_values(proposal,offsets=(-1,0,1,2,3))
            new=module._linear_score_decode(candidate).float();delta=new-old
            change=-2*(delta*gradient).sum(-1)+torch.einsum('oc,cd,od->o',delta,regularized[block],delta)
            accept=change<0
            actual=torch.where(accept[:,None],delta,torch.zeros_like(delta))
            b[:,begin:end].add_(actual);residual.sub_(x@actual.T)
            for name,value in params.items():
                current=value[:,block:block+1]
                current.copy_(torch.where(accept.reshape(-1,*([1]*(current.ndim-1))),candidate[name],current))
    return params
