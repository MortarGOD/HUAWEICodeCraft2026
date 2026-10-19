def _joint_quant_queries(hold, state, qh, kh, d, ix):
    # Query rows are independent in metadata selection, GPTQ and refinement.
    # Quantize only scored queries, keeping the ORIGINAL sequence's budget.
    x = dequantize_nvfp4(hold["q"][0][ix], hold["q"][1][ix]).float()
    if "qs" in state:
        x = _qks_apply_q(x, state["qs"].float(), qh, kh, d, inv=False)
    if "joint_transform" in state:
        t = state["joint_transform"].float().repeat_interleave(qh // kh, 0)
        x = (x.reshape(-1, qh, d).transpose(0, 1) @ t).transpose(0, 1).reshape(x.shape).contiguous()
    elif state.get("rot"):
        rot = _make_R(d, int(state.get("rseed") or 0))
        x = (x.reshape(-1, qh, d) @ rot).reshape(x.shape).contiguous()
    if "temperature_gain" in state:
        gain = state["temperature_gain"].float()
        x = (x.reshape(-1, qh, d) * gain[None, :, None]).reshape(x.shape).contiguous()
    p = _dyn_table(x, None, has_scale=False)
    unit = _params_unit_flat(p)
    values = _deq_params(p)
    if state.get("gq") == 1:
        u = state["u"].float().repeat_interleave(qh // state["u"].shape[0], 0)
        values = _gptq_quantize_batched(x.reshape(-1, qh, d).transpose(0, 1).contiguous(),
                                      unit.reshape(-1, qh, d).transpose(0, 1).contiguous(), u)
        values = values.transpose(0, 1).reshape(x.shape).contiguous()
    rows = hold["q"][0].shape[0]
    rf = int(state.get("rf") or 0)
    if rf and rows <= ATTN_REFINE_MAX_T and "H" in state:
        rounds = rf * (32 if rows <= 128 else (4 if rows>1024 else _REF_ATTN_ROUNDS))
        values = _refine_attn_heads(x, values, unit, state["H"], qh, rf,
                                    total_rounds=rounds)
    return _deq_params(_values_to_params(values, p))
