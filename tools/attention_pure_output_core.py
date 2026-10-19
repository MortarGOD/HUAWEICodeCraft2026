"""Replace mixed logit/output Gram with the output Gram alone."""
def _oj_state(st,h):
    metric=(h.float()+h.float().transpose(-1,-2))*.5
    if not bool(torch.isfinite(metric).all()) or not bool((metric.diagonal(dim1=-2,dim2=-1).mean(-1)>1.e-12).all()):
        return dict(st)
    upper=_upper_cholesky_inv(metric)
    if upper is None or not bool(torch.isfinite(upper).all()):return dict(st)
    answer=dict(st,H=metric.cpu().contiguous(),u=upper.cpu().contiguous())
    if isinstance(st.get('ts_fallback'),dict):answer['ts_fallback']=_oj_state(st['ts_fallback'],h)
    return answer
