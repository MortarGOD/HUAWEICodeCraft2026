"""Common local right rotation preserving parent maps and separate metrics."""


def _v437_rotation(d):
    h = torch.ones(1, 1)
    for _ in range(4):
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return torch.kron(torch.eye(d // 16), h * .25).contiguous()


def _v437_state(state, kh, d):
    if d % 16 or d > 512:
        return state
    if isinstance(state, dict) and state.get('post_rotation16'):
        return state
    r = _v437_rotation(d)
    old = state if isinstance(state, dict) else {}
    result = dict(old)
    if 'joint_transform' in old:
        transform = old['joint_transform'].float()
    else:
        transform = _make_R(d, int(old.get('rseed') or 0)) if old.get('rot') else None
        if transform is None:
            transform = torch.eye(d)
        transform = transform[None].repeat(kh, 1, 1)
    result.update(kvh=kh, rot=0, post_rotation16=True,
                  joint_transform=(transform @ r).contiguous())
    if isinstance(old.get('ts_fallback'), dict):
        result['ts_fallback'] = _v437_state(old['ts_fallback'], kh, d)
    if isinstance(old.get('H'), torch.Tensor):
        h = old['H'].double()
        transformed = r.double().T @ h @ r.double()
        result['H'] = ((transformed + transformed.transpose(-1, -2)) * .5).float().contiguous()
    if isinstance(old.get('u'), torch.Tensor):
        # QR transports the already damped inverse metric independently of H.
        u = torch.triu(old['u'].double())
        try:
            upper = torch.linalg.qr(u @ r.double(), mode='r').R
            signs = torch.where(upper.diagonal(dim1=-2, dim2=-1) < 0, -1., 1.)
            upper = upper * signs[..., :, None]
            if not bool(torch.isfinite(upper).all()) or bool((upper.diagonal(dim1=-2, dim2=-1) <= 0).any()):
                raise ValueError('invalid transported inverse factor')
            result['u'] = upper.float().contiguous()
            result['post_rotation_metric_fallback'] = False
        except (RuntimeError, ValueError):
            diagonal = u.square().sum(-2).mean(-1).clamp_min(1e-30).sqrt()
            result['u'] = (torch.eye(d).expand_as(u) * diagonal[..., None, None]).float().contiguous()
            result['post_rotation_metric_fallback'] = True
    return result
