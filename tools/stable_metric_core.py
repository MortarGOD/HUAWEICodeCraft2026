"""v402: reconstruct the missing Gram with an explicit triangular solve."""
_mr_parent_raw_metric = _joint_raw_metric


def _joint_raw_metric(st):
    if 'H' in st:
        return _mr_parent_raw_metric(st)
    u = st['u'].float()
    identity = torch.eye(u.shape[-1], dtype=u.dtype).expand_as(u)
    inverse_u = torch.linalg.solve_triangular(u, identity, upper=True)
    # U^T U = H_damped^-1, hence H_damped = U^-1 U^-T.
    # Do not reverse these two factors for a non-diagonal U.
    hd = inverse_u @ inverse_u.transpose(-1, -2)
    ridge = hd.diagonal(dim1=-2, dim2=-1).mean(-1) * (GPTQ_DAMP / (1 + GPTQ_DAMP))
    return (hd + hd.transpose(-1, -2)) * .5 - ridge[:, None, None] * torch.eye(hd.shape[-1])
