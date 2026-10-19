"""v403: same-size, mass-weighted current-query coreset for V kernel risk."""
_cq_parent_probabilities = _vk_probabilities


def _cq_representatives(f, kernel):
    n, width = f.shape
    count = min(_VK_QUERIES, n)
    centered = f - f.mean(0)
    covariance = centered.T @ centered / n
    metric = (kernel @ covariance @ kernel.transpose(-1, -2)).mean(0)
    eig, basis = torch.linalg.eigh((metric + metric.T) * .5)
    transform = basis * eig.clamp_min(0).sqrt()[None]
    points = f @ transform
    centers = [points.mean(0)]
    nearest = (points - centers[0]).square().sum(1)
    for _ in range(1,count):
        center = points[nearest.argmax()]
        centers.append(center)
        nearest = torch.minimum(nearest,(points-center).square().sum(1))
    centers = torch.stack(centers)
    norms = points.square().sum(1,keepdim=True)
    for _ in range(4):
        distances = norms + centers.square().sum(1)[None] - 2*points@centers.T
        assignment = distances.argmin(1)
        sizes = torch.zeros(count).scatter_add_(0,assignment,torch.ones(n))
        sums = torch.zeros(count,width).scatter_add_(0,assignment[:,None].expand(-1,width),f)
        representatives = sums / sizes.clamp_min(1)[:,None]
        centers = torch.where((sizes>0)[:,None],representatives@transform,centers)
    return representatives, sizes/n


def _vk_probabilities(x,state,kh):
    if x.shape[0]<=_VK_QUERIES or x.shape[0]>4096:
        return _cq_parent_probabilities(x,state,kh)
    try:
        f = _vk_features(x,state)
        kernel = state['kernel'].float()
        representatives,mass = _cq_representatives(f,kernel)
        logits = (representatives[None] @ kernel) @ f.T
        modes = state.get('query_sigma_modes')
        if modes is not None and bool(modes.any()):
            delta = (modes.float()@f.T)*(2.**.5)
            offsets = torch.cat((delta,-delta),1)
        else:
            offsets = torch.zeros(kernel.shape[0],1,x.shape[0])
        p = torch.softmax(logits[:,:,None,:]+offsets[:,None,:,:],-1)
        factors = p*(mass*representatives.shape[0]).sqrt()[None,:,None,None]
        if not bool(torch.isfinite(factors).all()):
            return _cq_parent_probabilities(x,state,kh)
        return factors.reshape(kh,-1,x.shape[0]).contiguous()
    except (RuntimeError,ValueError):
        return _cq_parent_probabilities(x,state,kh)
