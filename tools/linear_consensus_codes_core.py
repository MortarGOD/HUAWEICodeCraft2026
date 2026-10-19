"""Fixed-metadata weight coordinate updates with two calibration-fold descent."""
def _dc_refine(zs, ys, parent_params, budget=64):
    params = {k:v.clone() for k,v in parent_params.items()}
    b = _dequantize_linear_params(params).float()
    outputs, width = b.shape
    zfold = [torch.cat(zs[k::2]) for k in range(2)]
    yfold = [torch.cat(ys[k::2]) for k in range(2)]
    residual = [y-z@b.T for z,y in zip(zfold,yfold)]
    diagonal = [z.square().sum(0).clamp_min(1.e-30) for z in zfold]
    scale = (params['scale_factor'].float()*params['scale_lv2'].float()
             *params['scale_lv3'].float()*torch.ones_like(params['mant'],dtype=torch.float32)).reshape_as(b)/4
    sign=params['sign'].reshape_as(b)
    mant=params['mant'].reshape_as(b)
    codes=(sign.float()*mant.float()*4).round()
    gradient=sum(r.T@z for r,z in zip(residual,zfold))
    diag=diagonal[0]+diagonal[1]
    step=scale*torch.sign(gradient)
    score=2*gradient*step-diag[None,:]*step.square()
    order=score.clamp_min(0).sum(0).argsort(descending=True)[:min(budget,width)].tolist()
    changed=0
    for j in order:
        gradients=[r.T@z[:,j] for r,z in zip(residual,zfold)]
        pooled=gradients[0]+gradients[1]
        newcode=(codes[:,j]+torch.sign(pooled)).clamp(-7,7)
        delta=(newcode-codes[:,j])*scale[:,j]
        accepted=(delta!=0)
        for g,h in zip(gradients,diagonal):
            accepted &= 2*delta*g-delta.square()*h[j] > 0
        delta=torch.where(accepted,delta,torch.zeros_like(delta))
        newcode=torch.where(accepted,newcode,codes[:,j])
        for r,z in zip(residual,zfold):
            r.sub_(z[:,j,None]*delta[None,:])
        codes[:,j]=newcode
        sign[:,j]=torch.sign(newcode).to(sign.dtype)
        mant[:,j]=(newcode.abs()/4).to(mant.dtype)
        changed+=int(accepted.sum())
    DG_DEBUG['changed_weight_codes']=changed
    return params


_dg_parent_calibration = hif4_calibration_and_quantize_weight
DG_DEBUG = {}

@torch.inference_mode()
def hif4_calibration_and_quantize_weight(weight_quant, weight_scale, calib_activation_list):
    global DG_DEBUG
    parent = _dg_parent_calibration(weight_quant, weight_scale, calib_activation_list)
    state = parent['activation_state']
    outputs, inputs = weight_quant.shape
    DG_DEBUG = {'eligible': False, 'accepted': False}
    if outputs <= inputs or inputs > 4096 or len(calib_activation_list) < 5:
        return parent
    if 'unrestricted_target_weight' not in state:
        return parent
    DG_DEBUG['eligible'] = True
    weight = dequantize_nvfp4(weight_quant, weight_scale).float()
    b = _linear_score_decode(parent['weight_params']).float()
    zs, ys = [], []
    try:
        # Fit only the first sequences; parent calibration already saw all of
        # them, so the last sequence is only held out from this extra fit.
        for packed in calib_activation_list[:-1]:
            raw = dequantize_nvfp4(*packed).float()
            code = hif4_dynamic_quantize_activation(*packed, state)
            z = _linear_score_decode(code).float()
            y = raw @ weight.T
            n = min(64, raw.shape[0])
            idx = torch.linspace(0, raw.shape[0]-1, n).round().long()
            # Equal sequence/target-energy weighting (not an online score fit).
            factor = (n*y.square().mean()).clamp_min(1.e-30).rsqrt()
            zs.append(z[idx]*factor)
            ys.append(y[idx]*factor)
        params = _dc_refine(zs, ys, parent['weight_params'])
        if not DG_DEBUG['changed_weight_codes']:
            return parent
        candidate_state = dict(state)
        candidate_state.pop('unrestricted_continuous_projection', None)
        candidate_state.pop('linear_full_gram', None)
        _install_unrestricted_exact_target_state(
            candidate_state, state['unrestricted_target_weight'].float(), params)
        deployed = _dequantize_linear_params(params).float()
        if outputs >= 2*inputs:
            candidate_state['linear_full_gram'] = (deployed.T@deployed).contiguous()
        packed = calib_activation_list[-1]
        raw = dequantize_nvfp4(*packed).float()
        if raw.shape[0] < 2:
            return parent
        target = raw @ weight.T
        parent_code = hif4_dynamic_quantize_activation(*packed, state)
        candidate_code = hif4_dynamic_quantize_activation(*packed, candidate_state)
        old = _linear_score_decode(parent_code) @ _linear_score_decode(parent['weight_params']).T
        new = _linear_score_decode(candidate_code) @ _linear_score_decode(params).T
        old_loss = (target-old.float()).square().mean(-1)
        new_loss = (target-new.float()).square().mean(-1)
        DG_DEBUG['fold_ratios'] = [float(new_loss[i::2].mean()/old_loss[i::2].mean().clamp_min(1.e-30)) for i in range(2)]
        if all(bool(new_loss[i::2].mean() < old_loss[i::2].mean()*.9999) for i in range(2)):
            DG_DEBUG['accepted'] = True
            return {'weight_params': params, 'activation_state': candidate_state}
    except RuntimeError:
        DG_DEBUG['numerical_fallback'] = True
    return parent

