"""Short-sequence native V selection from the current calibration context."""


def _v440_eligible(t, kh, d):
    return 2 <= t <= 32 and d % 64 == 0 and kh * d <= 4096


def _v440_fit(samples, parent, qh, kh, d):
    usable = [s for s in samples if _v440_eligible(s['v'][0].shape[0], kh, d)][:3]
    if len(usable) < 2:
        return parent['v_state']
    ratios = []
    for sample in usable:
        raw = {r: dequantize_nvfp4(*sample[r]).float() for r in 'qkv'}
        q = _deq_params(hif4_dynamic_quantize_q(*sample['q'], qh, d, parent['q_state']))
        k = _deq_params(hif4_dynamic_quantize_k(*sample['k'], kh, d, parent['k_state']))
        old = _v440_parent_dynamic(*sample['v'], kh, d, parent['v_state'])
        native = _dyn_table(raw['v'], None, False)
        t = raw['v'].shape[0]
        p = _vr_probs(raw['q'], raw['k'], qh, kh, d).reshape(kh, -1, t)
        pq = _vr_probs(q, k, qh, kh, d).reshape(kh, -1, t)
        target = p @ raw['v'].reshape(t, kh, d).transpose(0, 1)
        errors = []
        for params in (old, native):
            v = _deq_params(params).reshape(t, kh, d).transpose(0, 1)
            errors.append((pq @ v - target).square().mean((1, 2)))
        ratios.append(errors[1] / errors[0].clamp_min(1e-30))
    ratio = torch.stack(ratios).mean(0)
    active = torch.isfinite(ratio) & (ratio < .99)
    if not bool(active.any()):
        return parent['v_state']
    return {'v440_parent': parent['v_state'], 'v440_native_heads': active.contiguous()}


_v440_parent_calibration = hif4_calibration_attention
_v440_parent_dynamic = hif4_dynamic_quantize_v


@torch.inference_mode()
def hif4_calibration_attention(samples, qh, kh, d):
    parent = _v440_parent_calibration(samples, qh, kh, d)
    return dict(parent, v_state=_v440_fit(samples, parent, qh, kh, d))


@torch.inference_mode()
def hif4_dynamic_quantize_v(quant, scale, kh, d, state):
    if not isinstance(state, dict) or 'v440_parent' not in state:
        return _v440_parent_dynamic(quant, scale, kh, d, state)
    old = state['v440_parent']
    if not _v440_eligible(quant.shape[0], kh, d):
        return _v440_parent_dynamic(quant, scale, kh, d, old)
    active = state['v440_native_heads'].bool()
    native = _dyn_table(dequantize_nvfp4(quant, scale).float(), None, False)
    if bool(active.all()):
        return native
    parent = _v440_parent_dynamic(quant, scale, kh, d, old)
    block_mask = active.repeat_interleave(d // 64)[None]
    return {key: torch.where(block_mask.reshape(1, -1, *([1] * (value.ndim - 2))),
                             native[key], value).contiguous() for key, value in parent.items()}
