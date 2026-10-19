"""Exact integer-sign H64 output decoder for direct-output Linear.

Each 64-channel carrier block uses S=8*H64, S S.T=64I.
Encode y with S.T/64 and decode with S; all decoder entries remain
exact HiF4 values, unlike a generic learned orthogonal decoder.
"""
_h64_parent_weight_params=_direct_output_weight_params


def _direct_output_weight_params(outputs,inputs,slots,decoder_scale):
    params=_h64_parent_weight_params(outputs,inputs,slots,decoder_scale)
    basis=torch.sign(_hadamard(torch.eye(64),64)).to(torch.bfloat16)
    rows=torch.arange(outputs)[:,None].expand_as(slots)
    valid=slots>=0
    r=rows[valid];block=slots[valid]//64;within=slots[valid]%64
    params['sign'].reshape(outputs,inputs//64,64)[r,block]=basis[within]
    params['mant'].reshape(outputs,inputs//64,64)[r,block]=1.
    return params
