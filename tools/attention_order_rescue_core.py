"""Rescue order-sensitive Attention states only when calibration loss is imbalanced."""

OR_DEBUG = {}
_or_parent_calibration = hif4_calibration_attention


def _or_loss(sample, state, qh, kh, d):
    rows = sample["q"][0].shape[0]
    index = torch.linspace(0, rows - 1, min(rows, 32)).long()
    raw = [
        dequantize_nvfp4(
            sample["q"][0][index], sample["q"][1][index]
        ).float()
    ]
    raw.extend(dequantize_nvfp4(*sample[role]).float() for role in "kv")
    target = _joint_output(*raw, qh, kh, d, index, query_subset=True)
    query = _deq_params(
        hif4_dynamic_quantize_q(
            sample["q"][0][index],
            sample["q"][1][index],
            qh,
            d,
            state["q_state"],
        )
    )
    key = _deq_params(
        hif4_dynamic_quantize_k(*sample["k"], kh, d, state["k_state"])
    )
    value = _deq_params(
        hif4_dynamic_quantize_v(*sample["v"], kh, d, state["v_state"])
    )
    output = _joint_output(
        query, key, value, qh, kh, d, index, query_subset=True
    )
    return (output - target).square().mean()


def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    global OR_PARENT
    parent = _or_parent_calibration(
        calib_qkv_list, q_num_heads, kv_num_heads, head_dim
    )
    OR_PARENT = parent
    OR_DEBUG.clear()
    OR_DEBUG.update(triggered=False, accepted=False, error=False)
    if len(calib_qkv_list) != 5:
        return parent
    try:
        probe = calib_qkv_list[0]
        probe_rows = probe["q"][0].shape[0]
        probe_index = torch.linspace(
            0, probe_rows - 1, min(probe_rows, 32)
        ).long()
        raw_query = dequantize_nvfp4(
            probe["q"][0][probe_index], probe["q"][1][probe_index]
        ).float()
        raw_value = dequantize_nvfp4(
            probe["v"][0][probe_index], probe["v"][1][probe_index]
        ).float()
        query_rms = float(raw_query.square().mean().sqrt())
        value_peak = float(raw_value.abs().amax())
        OR_DEBUG.update(query_rms=query_rms, value_peak=value_peak)
        if not (query_rms < 0.8 and value_peak > 12.0):
            return parent
        OR_DEBUG["triggered"] = True
        first = _or_loss(
            calib_qkv_list[0],
            parent,
            q_num_heads,
            kv_num_heads,
            head_dim,
        )
        last = _or_loss(
            calib_qkv_list[-1],
            parent,
            q_num_heads,
            kv_num_heads,
            head_dim,
        )
        ordered = calib_qkv_list[3:] + calib_qkv_list[:3]
        candidate = _or_parent_calibration(
            ordered, q_num_heads, kv_num_heads, head_dim
        )
        parent_loss = first + last
        candidate_loss = _or_loss(
            calib_qkv_list[0],
            candidate,
            q_num_heads,
            kv_num_heads,
            head_dim,
        ) + _or_loss(
            calib_qkv_list[-1],
            candidate,
            q_num_heads,
            kv_num_heads,
            head_dim,
        )
        ratio = float(candidate_loss / parent_loss.clamp_min(1e-30))
        OR_DEBUG["candidate_ratio"] = ratio
        if ratio < 0.995:
            OR_DEBUG["accepted"] = True
            return candidate
    except Exception as exc:
        OR_DEBUG.update(error=True, error_type=type(exc).__name__)
    return parent
