"""Deployed-code ALS with block GPTQ feedback; independent calibration probe."""
_dg_parent_calibration = hif4_calibration_and_quantize_weight
DG_DEBUG = {}


def _dg_fit(z, y, parent_b, ridge_ratio=.1):
    h = z.T @ z
    ridge = ridge_ratio * h.diagonal().mean().clamp_min(1.e-30)
    h.diagonal().add_(ridge)
    rhs = z.T @ y + ridge * parent_b.T
    proposal = torch.linalg.solve(h, rhs).T
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(h))
    upper = torch.linalg.cholesky(inverse, upper=True)
    return proposal, upper


def _dg_quantize(proposal, upper):
    work = proposal.clone()
    pieces = []
    width = work.shape[1]
    for start in range(0, width, 64):
        stop = start + 64
        original = work[:, start:stop]
        params = _quantize_linear_values(original, offsets=_LINEAR_WEIGHT_SCALE_OFFSETS)
        decoded = _dequantize_linear_params(params).float()
        pieces.append(params)
        if stop < width:
            # C_II^-1 C_IF = U_II^-1 U_IF for C=U.T U.
            error = torch.linalg.solve_triangular(
                upper[start:stop, start:stop].T,
                (original-decoded).T, upper=False).T
            work[:, stop:].sub_(error @ upper[start:stop, stop:])
    return {key: torch.cat([p[key] for p in pieces], dim=-4) for key in pieces[0]}


@torch.inference_mode()
def hif4_calibration_and_quantize_weight(weight_quant, weight_scale, calib_activation_list):
    global DG_DEBUG
    parent = _dg_parent_calibration(weight_quant, weight_scale, calib_activation_list)
    state = parent['activation_state']
    outputs, inputs = weight_quant.shape
    DG_DEBUG = {'eligible': False, 'accepted': False}
    if outputs <= inputs or inputs > 4096 or len(calib_activation_list) < 3:
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
        proposal, upper = _dg_fit(torch.cat(zs), torch.cat(ys), b)
        params = _dg_quantize(proposal, upper)
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
