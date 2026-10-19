"""Shared coordinates with independently transported refinement/GPTQ metrics."""


def _v438_state(state, role, kh, d):
    old = state if isinstance(state, dict) else {}
    if old.get('canonical_dual_metric'):
        return state
    rotation = _make_R(d, 0)
    if rotation is None:
        rotation = torch.eye(d)
    result = {k: v for k, v in old.items()
              if k not in ('qs', 'ks', 'joint_transform', 'rot', 'rseed', 'hc_matrix')}
    result.update(kvh=kh, rot=0, canonical_dual_metric=True,
                  joint_transform=rotation[None].repeat(kh, 1, 1))
    if isinstance(old.get('ts_fallback'), dict):
        result['ts_fallback'] = _v438_state(old['ts_fallback'], role, kh, d)
    plain = {k: v for k, v in old.items() if k != 'temperature_gain'}
    plain.setdefault('kvh', kh)
    a = _kt_input(torch.eye(d).repeat(1, kh), plain, kh, d, role)
    a = a.reshape(d, kh, d).transpose(0, 1).double()
    c = rotation.double()
    if isinstance(old.get('H'), torch.Tensor):
        h = old['H'].double()
        am = a.repeat_interleave(h.shape[0] // kh, 0)
        transformed = c.T @ am @ h @ am.transpose(-1, -2) @ c
        result['H'] = ((transformed + transformed.transpose(-1, -2)) * .5).float().contiguous()
    if isinstance(old.get('u'), torch.Tensor):
        u = torch.triu(old['u'].double())
        am = a.repeat_interleave(u.shape[0] // kh, 0)
        failed = False
        try:
            s = torch.linalg.solve(am, c.expand_as(am))
            upper = torch.linalg.qr(u @ s, mode='r').R
            signs = torch.where(upper.diagonal(dim1=-2, dim2=-1) < 0, -1., 1.)
            upper = upper * signs[..., :, None]
            if not bool(torch.isfinite(upper).all()) or bool((upper.diagonal(dim1=-2, dim2=-1) <= 0).any()):
                raise ValueError('invalid transported GPTQ factor')
            result['u'] = upper.float().contiguous()
        except (RuntimeError, ValueError):
            # Keep the shared coordinate contract even if a metric cannot move.
            result['u'] = torch.eye(d)[None].repeat(u.shape[0], 1, 1)
            failed = True
        result['canonical_dual_metric_fallback'] = failed
    return result
