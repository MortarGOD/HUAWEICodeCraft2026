"""Block-LDL activation proposal with a native hierarchical HiF4 quantizer."""

def _linear_ldl_proposal(values, factor):
    working=values.float().reshape(-1,values.shape[-1]).clone()
    upper=factor.float();width=working.shape[-1];pieces=[]
    for begin in range(0,width,64):
        end=begin+64
        params=_quantize_linear_values(working[:,begin:end],offsets=(0,1,2))
        reconstructed=_dequantize_linear_params(params).float()
        error=torch.linalg.solve_triangular(upper[begin:end,begin:end].T,
                    (working[:,begin:end]-reconstructed).T,upper=False).T
        if end<width:
            working[:,end:].sub_(error@upper[begin:end,end:])
        pieces.append(params)
    return {name:torch.cat([p[name] for p in pieces],dim=-4).reshape(
            *values.shape[:-1],width//64,*pieces[0][name].shape[2:]) for name in pieces[0]}
