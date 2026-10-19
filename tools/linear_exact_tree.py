"""L1 exact hierarchical HiF4 SSE reference and L4 finite-NVFP4 error tables.

255 positive E6M2 scales; ties prefer earlier SF, then lv2=1, lv3=1.
All mantissa * E6M2 products are exactly representable in BF16 (<=6 bits).
Float64 is used for error comparisons, not additional submitted precision.
"""
import torch


def scales():
    code = torch.arange(255)
    return torch.exp2((code // 4 - 48).double()) * (1. + (code % 4).double() / 4.)


def tree_choice(errors):
    # errors: [..., 3 products, 8 groups, 2 microgroups]
    e0, e1, e2 = errors.unbind(-3)
    low3 = (e1 < e0).long()
    high3 = (e2 < e1).long()
    low = torch.minimum(e0, e1).sum(-1)
    high = torch.minimum(e1, e2).sum(-1)
    high2 = high < low
    lv2 = 1 + high2.long()
    lv3 = 1 + torch.where(high2[..., None], high3, low3)
    return torch.minimum(low, high).sum(-1), lv2, lv3


def exact_tree(x, *, nvfp4=None):
    """General x or exact (E2M1 values, BF16 scales) table backend; no lossy key."""
    shape = x.shape
    flat = x.double().reshape(-1, 64)
    sf_all = scales()
    best_error = torch.full((flat.shape[0],), torch.inf, dtype=torch.float64)
    best_sf = torch.zeros(flat.shape[0], dtype=torch.long)
    best_l2 = torch.ones((flat.shape[0], 8), dtype=torch.long)
    best_l3 = torch.ones((flat.shape[0], 8, 2), dtype=torch.long)
    if nvfp4 is not None:
        quant, scale = nvfp4
        levels = torch.tensor([-6.,-4.,-3.,-2.,-1.5,-1.,-.5,0.,.5,1.,1.5,2.,3.,4.,6.])
        q = quant.float().reshape(-1, 64)
        indices = torch.searchsorted(levels, q)
        if not bool((indices < 15).all()) or not torch.equal(levels[indices], q):
            raise ValueError('NVFP4 table requires exact E2M1 input levels')
        sf16 = scale.reshape(-1,4).to(torch.bfloat16)
        unique, inv = torch.unique(sf16, return_inverse=True)
        # Includes absolute scale and BF16 multiplication, not just ratios.
        decoded = (unique[:, None] * levels.bfloat16()[None]).double()
        recovered = decoded[inv[...,None].expand(-1,-1,16), indices.reshape(-1,4,16)]
        if not torch.equal(recovered.reshape_as(flat), flat):
            raise ValueError('Transformed input is not the supplied NVFP4 tensor')
    for start in range(0, 255, 16):
        sfs = sf_all[start:start+16]
        unit = sfs[:,None] * torch.tensor([.25,.5,1.],dtype=torch.float64)[None]
        if nvfp4 is None:
            v = flat[:,None,None,:]
            step = unit[None,:,:,None]
            codes = (v/step).round().clamp(-7,7)
            errors = (codes*step-v).square().reshape(-1,len(sfs),3,8,2,4).sum(-1)
        else:
            v = decoded[:,None,None,:]
            step = unit[None,:,:,None]
            table = ((v/step).round().clamp(-7,7)*step-v).square()
            # [blocks, 4 NV blocks, SF, products, 16 element positions]
            gathered = table[inv[...,None].expand(-1,-1,16), :, :, indices.reshape(-1,4,16)]
            errors = gathered.permute(0,3,4,1,2).reshape(-1,len(sfs),3,8,2,4).sum(-1)
        loss,l2,l3 = tree_choice(errors)
        val, ix = loss.min(-1)
        take = val < best_error
        row = torch.arange(flat.shape[0])
        best_error = torch.where(take,val,best_error)
        best_sf = torch.where(take,ix+start,best_sf)
        best_l2 = torch.where(take[:,None],l2[row,ix],best_l2)
        best_l3 = torch.where(take[:,None,None],l3[row,ix],best_l3)
    base = sf_all[best_sf]
    step = base[:,None,None,None]*best_l2[:,:,None,None]*best_l3[:,:,:,None]/4
    code = (flat.reshape(-1,8,2,4)/step).round().clamp(-7,7)
    prefix = (*shape[:-1],shape[-1]//64)
    return {'scale_factor':base.reshape(*prefix,1,1,1).bfloat16(),
            'scale_lv2':best_l2.reshape(*prefix,8,1,1).bfloat16(),
            'scale_lv3':best_l3.reshape(*prefix,8,2,1).bfloat16(),
            'sign':code.sign().reshape(*prefix,8,2,4).bfloat16(),
            'mant':(code.abs()/4).reshape(*prefix,8,2,4).bfloat16()}
