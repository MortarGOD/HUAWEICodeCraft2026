"""Current-Query softmax-KL coordinate refinement using calibrated Key prototypes."""

DQ_DEBUG = {}
_dq_parent_calibration = hif4_calibration_attention
_dq_parent_q = hif4_dynamic_quantize_q


def _dq_keys(samples, parent, kh, d):
    pieces = []
    for sample in samples[:-1][-4:]:
        count = sample["k"][0].shape[0]
        index = torch.linspace(0, count - 1, min(count, 8)).long()
        raw = dequantize_nvfp4(
            sample["k"][0][index], sample["k"][1][index]
        ).float()
        key = _kt_input(raw, parent["k_state"], kh, d, "k")
        pieces.append(key.reshape(-1, kh, d).transpose(0, 1))
    return torch.cat(pieces, 1).contiguous()


def _dq_loss(value, keys, target, d):
    logits = torch.bmm(value, keys.transpose(1, 2)) / d ** 0.5
    return torch.logsumexp(logits, -1) - (target * logits).sum(-1)


def _dq_refine(q_quant, q_scale, qh, d, state, active=None):
    parent = _dq_parent_q(q_quant, q_scale, qh, d, state)
    rows = q_quant.shape[0]
    if rows <= 1 or rows > 1024:
        return parent
    raw = dequantize_nvfp4(q_quant, q_scale).float()
    source = _kt_input(raw, state, qh, d, "q")
    source = source.reshape(rows, qh, d).transpose(0, 1).contiguous()
    value = _deq_params(parent).reshape(rows, qh, d).transpose(0, 1).contiguous()
    unit = _params_unit_flat(parent).reshape(rows, qh, d).transpose(0, 1).contiguous()
    step = unit * 0.25
    keys = state["dq_key"].float()
    keys = keys.repeat_interleave(qh // keys.shape[0], 0)
    scaled_keys = keys / d ** 0.5
    target = torch.softmax(torch.bmm(source, scaled_keys.transpose(1, 2)), -1)
    if active is None:
        active_heads = torch.ones(qh, dtype=torch.bool)
    else:
        active_heads = active.repeat_interleave(qh // active.shape[0])
    for _ in range(2):
        logits = torch.bmm(value, scaled_keys.transpose(1, 2))
        probability = torch.softmax(logits, -1)
        old_loss = torch.logsumexp(logits, -1) - (target * logits).sum(-1)
        mean = torch.bmm(probability, scaled_keys)
        second = torch.bmm(probability, scaled_keys.square()) - mean.square()
        gradient = torch.bmm(probability - target, scaled_keys)
        v4 = torch.round(value / step)
        up_cost = gradient * step + 0.5 * second * step.square()
        down_cost = -gradient * step + 0.5 * second * step.square()
        up_cost.masked_fill_(v4 >= 7, _ROUND_INF)
        down_cost.masked_fill_(v4 <= -7, _ROUND_INF)
        costs = torch.cat((up_cost, down_cost), -1)
        best_cost, choice = costs.min(-1)
        coordinate = choice.remainder(d)
        direction = torch.where(choice < d, 1.0, -1.0)
        delta = direction * step.gather(-1, coordinate.unsqueeze(-1)).squeeze(-1)
        proposal = value.clone()
        proposal.scatter_add_(-1, coordinate.unsqueeze(-1), delta.unsqueeze(-1))
        new_loss = _dq_loss(proposal, keys, target, d)
        accept = (new_loss < old_loss) & (best_cost < 0) & active_heads[:, None]
        value = torch.where(accept[:, :, None], proposal, value)
    flattened = value.transpose(0, 1).reshape(rows, -1).contiguous()
    return _values_to_params(flattened, parent)


def _dq_guard(samples, parent, qh, kh, d):
    DQ_DEBUG.clear()
    DQ_DEBUG.update(accepted=0, error=False)
    if len(samples) < 3 or d % 64:
        return parent
    hold = samples[-1]
    rows = hold["q"][0].shape[0]
    if not 4 <= rows <= 4096:
        return parent
    proposed = dict(parent["q_state"], dq_key=_dq_keys(samples, parent, kh, d))
    index = torch.linspace(0, rows - 1, min(rows, 64)).long()
    raw = [
        dequantize_nvfp4(hold["q"][0][index], hold["q"][1][index]).float()
    ]
    raw.extend(dequantize_nvfp4(*hold[role]).float() for role in "kv")
    target = _joint_output(*raw, qh, kh, d, index, query_subset=True)
    old_query = _deq_params(
        _dq_parent_q(hold["q"][0][index], hold["q"][1][index], qh, d, parent["q_state"])
    )
    trial_query = _deq_params(
        _dq_refine(
            hold["q"][0][index],
            hold["q"][1][index],
            qh,
            d,
            proposed,
            torch.ones(kh, dtype=torch.bool),
        )
    )
    key = _deq_params(hif4_dynamic_quantize_k(*hold["k"], kh, d, parent["k_state"]))
    value = _deq_params(hif4_dynamic_quantize_v(*hold["v"], kh, d, parent["v_state"]))

    def cost(query):
        error = (
            _joint_output(query, key, value, qh, kh, d, index, query_subset=True)
            - target
        ).square().mean((-1, -2))
        return torch.stack((error[::2].mean(0), error[1::2].mean(0)))

    old = cost(old_query)
    new = cost(trial_query)
    ratio = new / old.clamp_min(1e-30)
    take = (ratio < 0.995).all(0)
    DQ_DEBUG.update(
        accepted=int(take.sum()),
        ratio=ratio.mean(0).tolist(),
        worst=ratio.amax(0).tolist(),
    )
    if not bool(take.any()):
        return parent
    return dict(parent, q_state=dict(proposed, dq_active=take))


def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    parent = _dq_parent_calibration(
        calib_qkv_list, q_num_heads, kv_num_heads, head_dim
    )
    try:
        return _dq_guard(
            calib_qkv_list, parent, q_num_heads, kv_num_heads, head_dim
        )
    except Exception as exc:
        DQ_DEBUG.update(error=True, error_type=type(exc).__name__)
        return parent


def hif4_dynamic_quantize_q(q_quant, q_scale, q_num_heads, head_dim, q_state):
    if (
        not isinstance(q_state, dict)
        or "dq_active" not in q_state
        or not 4 <= q_quant.shape[0] <= 1024
    ):
        return _dq_parent_q(q_quant, q_scale, q_num_heads, head_dim, q_state)
    return _dq_refine(
        q_quant,
        q_scale,
        q_num_heads,
        head_dim,
        q_state,
        q_state["dq_active"],
    )
