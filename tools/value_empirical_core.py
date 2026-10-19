"""v401: weighted empirical Q residual representatives for V risk."""
_VE_CENTERS = 8
_ve_parent_fit = _vk_fit
_ve_parent_probabilities = _vk_probabilities


def _ve_coefficients(f, q, k, groups, qh, kh, d):
    n, width = f.shape
    h = f.T @ f
    fq = f.T @ q
    eye = torch.eye(width)
    bk = torch.cholesky_solve(f.T @ k, torch.linalg.cholesky(h + eye * (.03*n)))
    bk = bk.reshape(width, kh, d).permute(1,0,2).repeat_interleave(qh//kh,0)
    residual = torch.empty_like(q)
    for start, stop in groups:
        fs = f[start:stop]
        hh = h - fs.T @ fs + eye * (.03*(n-stop+start))
        bq = torch.cholesky_solve(fq-fs.T@q[start:stop],torch.linalg.cholesky(hh))
        residual[start:stop] = q[start:stop] - fs@bq
    coefficients = residual.reshape(n,qh,d).transpose(0,1) @ bk.transpose(1,2) / (d**.5)
    coefficients[:,:,-1] = 0  # softmax-null constant logit direction
    return coefficients.contiguous()


def _ve_centroids(coefficients, scale):
    heads, n, width = coefficients.shape
    count = min(_VE_CENTERS, n)
    points = coefficients * scale
    cent = points.mean(1,keepdim=True)
    nearest = (points-cent).square().sum(-1)
    seeds = [cent[:,0]]
    for _ in range(1,count):
        index = nearest.argmax(1)
        item = points[torch.arange(heads),index]
        seeds.append(item)
        nearest = torch.minimum(nearest,(points-item[:,None]).square().sum(-1))
    cent = torch.stack(seeds,1)
    norm = points.square().sum(-1,keepdim=True)
    for _ in range(4):
        distances = norm + cent.square().sum(-1)[:,None] - 2*torch.bmm(points,cent.transpose(1,2))
        assignment = distances.argmin(-1)
        sizes = torch.zeros(heads,count).scatter_add_(1,assignment,torch.ones(heads,n))
        sums = torch.zeros(heads,count,width).scatter_add_(1,assignment[:,:,None].expand(-1,-1,width),coefficients)
        representatives = sums / sizes.clamp_min(1)[:,:,None]
        cent = torch.where((sizes>0)[:,:,None],representatives*scale,cent)
    # These means and masses share the same assignment: their weighted mean
    # equals the empirical mean before BF16 packing, even with empty clusters.
    return representatives, sizes / n


def _vk_fit(samples, qh, kh, d):
    if len(samples)<3:
        return _ve_parent_fit(samples,qh,kh,d)
    state = _vu_parent_fit(samples,qh,kh,d)
    try:
        arrays={r:[] for r in 'qkv'};groups=[];count=0
        for sample in samples[:-1]:
            t=sample['v'][0].shape[0]
            ix=torch.linspace(0,t-1,min(256,t)).long()
            groups.append((count,count+ix.numel()));count+=ix.numel()
            for role in 'qkv':arrays[role].append(dequantize_nvfp4(*sample[role]).float()[ix])
        q,k,v=[torch.cat(arrays[r]) for r in 'qkv']
        f=_vk_features(v,state)
        coefficients=_ve_coefficients(f,q,k,groups,qh,kh,d)
        scale=f.var(0,unbiased=False).clamp_min(1e-12).sqrt()
        centers,mass=_ve_centroids(coefficients,scale)
        centers=centers.to(torch.bfloat16).contiguous()
        if not bool(torch.isfinite(centers).all() and torch.isfinite(mass).all()):
            return _ve_parent_fit(samples,qh,kh,d)
        state['query_residual_centers']=centers
        state['query_residual_mass']=mass.contiguous()
        return state
    except (RuntimeError,ValueError):
        return _ve_parent_fit(samples,qh,kh,d)


def _vk_probabilities(x,state,kh):
    centers=state.get('query_residual_centers')
    if centers is None or x.shape[0]>4096:
        return _ve_parent_probabilities(x,state,kh)
    f=_vk_features(x,state)
    ix=torch.linspace(0,x.shape[0]-1,min(_VK_QUERIES,x.shape[0])).long()
    logits=(f[ix].unsqueeze(0)@state['kernel'].float())@f.T
    offsets=centers.float()@f.T
    p=torch.softmax(logits[:,:,None,:]+offsets[:,None,:,:],-1)
    # Gram factors, not normalized probability rows. Mean_R ||factor_R e||^2
    # is the mass-weighted risk across heads, queries and representatives.
    weights=(state['query_residual_mass']*centers.shape[1]).sqrt()
    return (p*weights[:,None,:,None]).reshape(kh,-1,x.shape[0]).contiguous()
