"""Use double intermediates only for joint-transform construction."""
def stable_joint_source(source):
    begin=source.index('def _joint_balanced_states(')
    end=source.index('\ndef _joint_quant_queries(',begin)
    block=source[begin:end]
    block=block.replace('hk = _joint_raw_metric(states["q_state"])','hk = _joint_raw_metric(states["q_state"]).double()')
    block=block.replace('hq = _joint_raw_metric(states["k_state"])','hq = _joint_raw_metric(states["k_state"]).double()')
    block=block.replace('rot @ transform if rot is not None else transform','rot.double() @ transform if rot is not None else transform')
    block=block.replace('joint_transform=full.contiguous(), u=inv.contiguous()','joint_transform=full.float().contiguous(), u=inv.float().contiguous()')
    assert '.double()' in block and 'u=inv.float()' in block
    return source[:begin]+block+source[end:]


def triangular_metric_source(source):
    needle='    hd = torch.cholesky_inverse(st["u"].float(), upper=True)'
    assert source.count(needle)==1
    replacement='''    upper = st["u"].float().contiguous()
    identity = torch.eye(upper.shape[-1], dtype=upper.dtype).expand(upper.shape)
    inverse = torch.linalg.solve_triangular(upper, identity, upper=True)
    hd = inverse @ inverse.transpose(-1, -2)'''
    return source.replace(needle,replacement)
