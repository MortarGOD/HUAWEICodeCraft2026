"""Full-channel calibration V->K regression and compressed query logits."""
import torch


def fit_key_predictor(samples,ridge=.01,max_rows=512):
    if not samples or ridge<=0 or max_rows<=0:
        raise ValueError('Nonempty samples and positive ridge/row budget required')
    vs=[];ks=[]
    identity=all(torch.equal(s['v'],s['k']) for s in samples)
    for sample in samples:
        t=sample['v'].shape[0]
        if t<1 or sample['k'].shape[0]!=t:raise ValueError('Nonempty aligned K/V required')
        ix=torch.linspace(0,t-1,min(t,max_rows)).long()
        vs.append(sample['v'][ix].float());ks.append(sample['k'][ix].float())
    v=torch.cat(vs);k=torch.cat(ks);c=v.shape[1]
    if identity:
        return dict(identity=True,center=torch.zeros(c),scale=torch.ones(c),
                    coef=torch.eye(c),bias=torch.zeros(c))
    center=v.mean(0);scale=(v-center).square().mean(0).sqrt()
    scale=scale.clamp_min(scale.mean().clamp_min(1e-12)*1e-4)
    f=(v-center)/scale;bias=k.mean(0)
    gram=f.T@f+torch.eye(c)*(ridge*f.shape[0])
    coef=torch.cholesky_solve(f.T@(k-bias),torch.linalg.cholesky(gram))
    return dict(identity=False,center=center.contiguous(),scale=scale.contiguous(),
                coef=coef.contiguous(),bias=bias.contiguous())


def predict_keys(v,state):
    if state['identity']:return v.float()
    return ((v.float()-state['center'])/state['scale'])@state['coef']+state['bias']


def fuse_queries(state,queries,kh,d):
    """queries [Qheads,N,d]; deploy logits directly, no current K required."""
    qh=queries.shape[0]
    if qh%kh or queries.shape[-1]!=d or state['coef'].shape[1]!=kh*d:
        raise ValueError('Invalid grouped-query dimensions')
    basis=state['coef'].reshape(-1,kh,d).permute(1,0,2).repeat_interleave(qh//kh,0)
    bias=state['bias'].reshape(kh,d).repeat_interleave(qh//kh,0)
    anchors=(queries@basis.transpose(1,2))/d**.5
    offset=(queries*bias[:,None,:]).sum(-1)/d**.5
    return dict(center=state['center'],scale=state['scale'],anchors=anchors.contiguous(),
                offset=offset.contiguous(),heads=qh,kv_heads=kh)


def query_probabilities(v,state):
    f=(v.float()-state['center'])/state['scale']
    logits=state['anchors']@f.T+state['offset'][:,:,None]
    p=torch.softmax(logits,-1)
    return p.reshape(state['kv_heads'],-1,v.shape[0]).contiguous()
