"""Candidate v137: guarded calibration-selected V mantissa rounding.

Linear is byte-for-byte v120 and Attention keeps v123's restored v10 RRS path.
The E6 offset-4 tree had zero selections across all 112,800 Q/K/V blocks in the
50-case fitted set, so dynamic and calibration quantization search offsets
-1..3 only.  After selecting the RRS Q/K transform, calibration output MSE
chooses exact ties-to-even RTN or one V threshold from 0.375/0.5/0.625.  A
negative scalar encodes exact RTN, so every calibration group can fall back to
the v124 path and dynamic quantization retains the same candidate count.

Linear keeps calibration-only transforms, residual-feedback Weight encoding,
layout guards, and alternating candidate selection.  Online Q(A) fitting uses
only projections and diagonal residuals recomputed from the final returned
HiF4 Q(W); no original-W projection, cross term, or recoverable Weight target
is stored in activation_state.  Calibration compares v29's exact compressed
Q(W) metric with a 5% isotropic diagonal trust region using true A@W outputs
over all five layouts; online fitting uses only the selected Q(W)-derived metric.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch


E6M2_MIN = 2.0 ** (-48)
E6M2_MAX = 49152.0  # 2^15 * 1.5

_LINEAR_SCALE_OFFSETS = (0, 1, 2)
_LINEAR_WIDE_SCALE_OFFSETS = (0, 1, 2, 3, 4)
_LINEAR_WEIGHT_SCALE_OFFSETS = (0, 1, 2, 3)
_LINEAR_WEIGHT_SATURATION_OFFSETS = (-1, 0, 1, 2, 3)
_LINEAR_PROFILE_MIN_GAIN = 0.005
_LINEAR_PROFILE_ROWS_PER_SHAPE = 64
_LINEAR_HADAMARD = 64
_LINEAR_BALANCE_EXPONENT = 0.44
_LINEAR_POWER_ITERS = 3
_LINEAR_OUTPUT_RANK = 4
_LINEAR_GLOBAL_OUTPUT_RANK = 2048
_LINEAR_GLOBAL_SCALE_OFFSETS = (0, 1, 2, 3, 4, 5, 6, 7)
_LINEAR_GLOBAL_JOINT_SWEEPS = 4
_LINEAR_GLOBAL_MULTI_SWEEP_MAX_ROWS = 512
_LINEAR_GLOBAL_ROW_CHUNK = 64
_LINEAR_GLOBAL_MIN_COMPRESSION = 32
_LINEAR_GLOBAL_SELECT_ROWS_PER_SHAPE = 16
_LINEAR_GLOBAL_SELECT_MIN_GAIN = 0.003
_LINEAR_GLOBAL_JOINT_MAX_LAYOUT_REGRESSION = 1.10
_LINEAR_GLOBAL_JOINT_MIN_WIN_LAYOUTS = 3
_LINEAR_GLOBAL_RIDGE_RATIO = 0.05
_LINEAR_ALTERNATING_MIN_GAIN = 0.005
_LINEAR_ALTERNATING_MAX_LAYOUT_REGRESSION = 1.05
_LINEAR_ALTERNATING_MIN_WIN_LAYOUTS = 3
_LINEAR_FEEDBACK_ROWS = 32
_LINEAR_FEEDBACK_WINDOW = 32
_LINEAR_FEEDBACK_RIDGE = 0.1
_LINEAR_FEEDBACK_STEP = 0.75
_LINEAR_FEEDBACK_ENERGY_ORDER = True
_LINEAR_FEEDBACK_DESCENDING = True

_ATTENTION_QK_OFFSETS = (0, 1, 2, 3, 4)
_ATTENTION_V_OFFSETS = (-1, 0, 1, 2, 3)
_ATTENTION_ENERGY_PENALTY = 0.125
_ATTENTION_NEAR_ZERO_RATIO = 0.01
_ATTENTION_SPARSE_THRESHOLD = 0.25


# =============================================================================
# NVFP4 helper
# =============================================================================

def dequantize_nvfp4(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    blk_size: int = 16,
) -> torch.Tensor:
    """Decode NVFP4 carrier + block scale into a BF16 tensor."""
    channels = int(quant_float.shape[-1])
    if channels % blk_size != 0:
        raise ValueError(
            f"last dimension {channels} is not divisible by block size {blk_size}"
        )

    x = quant_float.unflatten(-1, (-1, blk_size))
    x = x * scale_float.unsqueeze(-1)
    return x.flatten(-2, -1).to(torch.bfloat16)


# =============================================================================
# HiF4 conversion helpers
# =============================================================================

def _e6m2_quant(x: torch.Tensor) -> torch.Tensor:
    """Round a positive scale into the legal unsigned E6M2 value set."""
    x = torch.clamp(x, min=E6M2_MIN, max=E6M2_MAX)
    exp = torch.floor(torch.log2(x))
    mant = torch.round(x * (2.0 ** (2.0 - exp)))
    return mant * (2.0 ** (exp - 2.0))


def _s1p2_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a signed value into HiF4 S1P2 sign and mantissa fields."""
    sign = torch.sign(x)
    mag = torch.abs(x)
    q = torch.round(mag * 4.0) / 4.0
    q = torch.clamp(q, min=0.0, max=1.75)
    sign = torch.where(q == 0.0, torch.zeros_like(sign), sign)
    return sign, q


def _e6m2_candidates(sf: torch.Tensor, radius: int = 1) -> list[torch.Tensor]:
    """Return a small list of legal E6M2 candidates around ``sf``."""
    base_exp = torch.floor(torch.log2(sf))
    candidates: list[torch.Tensor] = []
    for delta in range(-radius, radius + 1):
        exp = (base_exp + delta).clamp(-48.0, 15.0)
        for mant in (1.0, 1.25, 1.5, 1.75):
            value = (2.0 ** exp) * mant
            value = torch.clamp(value, min=E6M2_MIN, max=E6M2_MAX)
            candidates.append(value)
    return candidates


def _hif4_from_scale(
    xb: torch.Tensor,
    v16: torch.Tensor,
    v8: torch.Tensor,
    scale_factor: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Quantize a reshaped block tensor given a candidate E6M2 scale factor."""
    rec = 1.0 / scale_factor

    e1_8 = (v8 * rec.unsqueeze(-1) >= 4.0).to(torch.float32)
    e1_8_pairs = e1_8.unsqueeze(-1)
    e1_16 = (
        v16
        * rec.unsqueeze(-1).unsqueeze(-1)
        * (2.0 ** (-e1_8_pairs))
        >= 2.0
    ).to(torch.float32)

    e1_8_for = e1_8.unsqueeze(-1).unsqueeze(-1)
    e1_16_for = e1_16.unsqueeze(-1)
    element_scale = (
        scale_factor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        * (2.0 ** e1_8_for)
        * (2.0 ** e1_16_for)
    )
    scaled = xb / element_scale
    sign, mant = _s1p2_quant(scaled)
    dequant = sign * mant * element_scale

    params = {
        "scale_factor": scale_factor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
        "scale_lv2": (2.0 ** e1_8).unsqueeze(-1).unsqueeze(-1),
        "scale_lv3": (2.0 ** e1_16).unsqueeze(-1),
        "sign": sign,
        "mant": mant,
    }
    return params, dequant


def _quantize_hif4(
    x: torch.Tensor,
    importance: torch.Tensor | None = None,
    search_candidates: bool = False,
    candidate_radius: int = 1,
) -> dict[str, torch.Tensor]:
    """Convert a BF16/FP32 tensor to HiF4 parameters along its last dim."""
    x = x.to(torch.float32)
    shape = x.shape
    channels = int(shape[-1])
    if channels % 64 != 0:
        raise ValueError(
            f"last dimension {channels} is not divisible by HiF4 block size 64"
        )

    num_blocks = channels // 64
    prefix = shape[:-1]
    xb = x.reshape(*prefix, num_blocks, 8, 2, 4)

    v16 = xb.abs().amax(dim=-1)
    v8 = v16.amax(dim=-1)
    vmax = v8.amax(dim=-1, keepdim=True)
    sf_std = _e6m2_quant(vmax.squeeze(-1) / 7.0)

    if not search_candidates:
        params, _ = _hif4_from_scale(xb, v16, v8, sf_std)
        return params

    candidates = _e6m2_candidates(sf_std, radius=candidate_radius)
    all_params: list[dict[str, torch.Tensor]] = []
    costs: list[torch.Tensor] = []

    if importance is not None:
        imp_b = importance.to(torch.float32)
    else:
        imp_b = None

    for sf in candidates:
        params, dequant = _hif4_from_scale(xb, v16, v8, sf)
        err = (dequant - xb) ** 2
        if imp_b is not None:
            cost = (err * imp_b).sum(dim=(-3, -2, -1))
        else:
            cost = err.sum(dim=(-3, -2, -1))
        all_params.append(params)
        costs.append(cost)

    cost_stack = torch.stack(costs, dim=0)  # [C, *prefix, nblocks]
    best_idx = torch.argmin(cost_stack, dim=0)  # [*prefix, nblocks]

    result: dict[str, torch.Tensor] = {}
    for key in all_params[0]:
        stacked = torch.stack([p[key] for p in all_params], dim=0)
        tail_shape = stacked.shape[1 + len(prefix) + 1:]
        index = best_idx.reshape((1,) + best_idx.shape + (1,) * len(tail_shape))
        index = index.expand((1,) + best_idx.shape + tail_shape)
        result[key] = torch.gather(stacked, 0, index).squeeze(0)
    return result


def _dequantize_hif4_params(
    params: dict[str, torch.Tensor],
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Decode HiF4 parameters back to a floating-point tensor."""
    return (
        params["sign"].to(torch.float32)
        * params["mant"].to(torch.float32)
        * params["scale_lv3"].to(torch.float32)
        * params["scale_lv2"].to(torch.float32)
        * params["scale_factor"].to(torch.float32)
    ).reshape(shape)


def _decode_e6m2(code: torch.Tensor) -> torch.Tensor:
    """Decode the compact unsigned E6M2 code used by the Linear search."""
    code64 = code.to(torch.int64)
    mantissa = 1.0 + (code64 & 3).to(torch.float32) / 4.0
    return torch.ldexp(mantissa, (code64 >> 2) - 48)


def _encode_e6m2(scale: torch.Tensor) -> torch.Tensor:
    """Encode a positive scale as a compact legal unsigned E6M2 code."""
    positive = scale > 0
    safe = torch.where(positive, scale, torch.ones_like(scale))
    exponent = torch.floor(torch.log2(safe)).to(torch.int64)
    normalized = torch.ldexp(safe, -exponent)
    mantissa = torch.round((normalized - 1.0) * 4.0).to(torch.int64)
    carry = mantissa == 4
    exponent = (exponent + carry.to(torch.int64)).clamp(-48, 15)
    mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa).clamp(0, 3)
    code = (((exponent + 48) << 2) | mantissa).clamp(0, 0xFE)
    default = torch.full_like(code, 48 << 2)
    return torch.where(positive, code, default).to(torch.uint8)


def _best_linear_for_base(
    groups: torch.Tensor,
    scale_code: torch.Tensor,
    weights: torch.Tensor | None = None,
    rounding_threshold: float = 0.5,
) -> tuple[torch.Tensor, ...]:
    """Find the best legal E1_8/E1_16 split for one Linear base scale."""
    base = _decode_e6m2(scale_code)
    factors = groups.new_tensor((1.0, 2.0, 4.0)).view(
        *([1] * (groups.ndim - 2)), 3, 1, 1
    )
    scaled = groups.unsqueeze(-3) / base.unsqueeze(-3) / factors
    negative = scaled < 0
    units = scaled.abs().clamp(max=1.75) * 4.0
    if rounding_threshold == 0.5:
        mantissa = torch.round(units).to(torch.uint8).clamp(max=7)
    else:
        lower = torch.floor(units)
        mantissa = (
            lower + ((units - lower) >= rounding_threshold)
        ).to(torch.uint8).clamp(max=7)
    reconstructed = (
        base.unsqueeze(-3)
        * factors
        * torch.where(negative, -1.0, 1.0)
        * mantissa.to(torch.float32)
        / 4.0
    )
    errors = (groups.unsqueeze(-3) - reconstructed).square()
    if weights is not None:
        errors = errors * weights.unsqueeze(-3)
    errors = errors.sum(dim=-1)
    error0, lv30 = errors[..., :2, :].min(dim=-2)
    error1, lv31 = errors[..., 1:, :].min(dim=-2)
    use1 = error1.sum(dim=-1) < error0.sum(dim=-1)
    lv2 = use1.to(torch.uint8)
    lv3 = torch.where(use1.unsqueeze(-1), lv31, lv30).to(torch.uint8)
    factor_index = (lv2.unsqueeze(-1) + lv3).to(torch.int64)
    index = factor_index.unsqueeze(-2).unsqueeze(-1).expand(
        *factor_index.shape[:-1], 1, 2, 4
    )
    selected = torch.gather(reconstructed, -3, index).squeeze(-3)
    selected_errors = (groups - selected).square()
    if weights is not None:
        selected_errors = selected_errors * weights
    score = selected_errors.sum(dim=(-3, -2, -1))
    return (
        score,
        scale_code,
        lv2.unsqueeze(-1).unsqueeze(-1),
        lv3.unsqueeze(-1),
        torch.gather(negative, -3, index).squeeze(-3),
        torch.gather(mantissa, -3, index).squeeze(-3),
    )


def _quantize_linear_values(
    values: torch.Tensor,
    importance: torch.Tensor | None = None,
    offsets: Iterable[int] = _LINEAR_SCALE_OFFSETS,
    projection: torch.Tensor | None = None,
    projection_strength: torch.Tensor | None = None,
    rounding_threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Quantize Linear values using v138's compact per-block scale search."""
    channels = int(values.shape[-1])
    groups = values.to(torch.float32).reshape(
        *values.shape[:-1], channels // 64, 8, 2, 4
    )
    weights = None
    if importance is not None:
        weights = importance.to(torch.float32).reshape(
            1, channels // 64, 8, 2, 4
        )
    peak = groups.abs().amax(dim=(-3, -2, -1), keepdim=True) / 7.0
    peak_code = _encode_e6m2(peak)
    best = None
    for offset in offsets:
        code = (
            peak_code.to(torch.int16).add(int(offset)).clamp(0, 0xFE).to(torch.uint8)
        )
        current = _best_linear_for_base(
            groups, code, weights, rounding_threshold=rounding_threshold
        )
        if projection is not None and projection_strength is not None:
            _, selected_code, lv2, lv3, negative, mantissa_code = current
            reconstructed = (
                _decode_e6m2(selected_code)
                * (1.0 + lv2.to(torch.float32))
                * (1.0 + lv3.to(torch.float32))
                * torch.where(negative, -1.0, 1.0)
                * mantissa_code.to(torch.float32)
                / 4.0
            )
            residual = groups - reconstructed
            prefix_dims = groups.ndim - 4
            rank = int(projection.shape[0])
            project = projection.to(torch.float32).reshape(
                rank, *([1] * prefix_dims), *projection.shape[1:]
            )
            strength = projection_strength.to(torch.float32).reshape(
                rank,
                *([1] * prefix_dims),
                projection_strength.shape[-1],
            )
            projected = (residual.unsqueeze(0) * project).sum(
                dim=(-3, -2, -1)
            )
            current = (
                current[0] + (strength * projected.square()).sum(dim=0),
            ) + current[1:]
        if best is None:
            best = current
            continue
        choose = current[0] < best[0]
        mask = choose[..., None, None, None]
        best = (torch.where(choose, current[0], best[0]),) + tuple(
            torch.where(mask, new, old)
            for new, old in zip(current[1:], best[1:])
        )

    if best is None:
        raise RuntimeError("Linear scale candidate set must not be empty")
    _, code, lv2, lv3, negative, mantissa_code = best
    mantissa = mantissa_code.to(torch.bfloat16) / 4.0
    sign = torch.where(
        mantissa == 0,
        torch.zeros_like(mantissa),
        torch.where(
            negative,
            -torch.ones_like(mantissa),
            torch.ones_like(mantissa),
        ),
    )
    return {
        "scale_factor": _decode_e6m2(code).to(torch.bfloat16),
        "scale_lv2": 1.0 + lv2.to(torch.bfloat16),
        "scale_lv3": 1.0 + lv3.to(torch.bfloat16),
        "sign": sign,
        "mant": mantissa,
    }


def _dequantize_linear_params(params: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        params["scale_factor"].to(torch.float32)
        * params["scale_lv2"].to(torch.float32)
        * params["scale_lv3"].to(torch.float32)
        * params["sign"].to(torch.float32)
        * params["mant"].to(torch.float32)
    ).flatten(-4, -1)


def _global_output_cost_state(
    quantized_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a high-rank output metric using only the final fixed Q(W)."""
    work = quantized_weight.to(torch.float32)
    rank = min(
        _LINEAR_GLOBAL_OUTPUT_RANK,
        int(work.shape[0]),
        int(work.shape[1]),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260823)
        if work.shape[0] >= work.shape[1]:
            if rank == work.shape[1]:
                basis = torch.linalg.qr(work).Q
            else:
                random = torch.randn(
                    work.shape[1], rank, dtype=work.dtype, device=work.device
                )
                basis = torch.linalg.qr(work @ random).Q
                basis = torch.linalg.qr(work.T @ basis).Q
                basis = torch.linalg.qr(work @ basis).Q
            quantized_projection = basis.T @ work
        elif rank == work.shape[0]:
            quantized_projection = work
        else:
            random = torch.randn(
                work.shape[1], rank, dtype=work.dtype, device=work.device
            )
            basis = torch.linalg.qr(work @ random).Q
            basis = torch.linalg.qr(work.T @ basis).Q
            basis = torch.linalg.qr(work @ basis).Q
            quantized_projection = basis.T @ work
    quantized_diagonal = (
        work.square().sum(0)
        - quantized_projection.square().sum(0)
    ).clamp_min(0.0)
    return quantized_diagonal, quantized_projection


def _quantize_linear_qw_single_sweep(
    values: torch.Tensor,
    diagonal: torch.Tensor,
    projection: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Minimize ||(A-Q(A))Q(W).T|| with one sequential block sweep."""
    candidates = [
        _quantize_linear_values(
            values,
            diagonal.reshape(-1, 8, 2, 4),
            offsets=(offset,),
        )
        for offset in _LINEAR_GLOBAL_SCALE_OFFSETS
    ]
    reconstructed = [_dequantize_linear_params(params) for params in candidates]
    block_count = int(values.shape[-1]) // 64
    prefix = values.shape[:-1]
    initial_index = _LINEAR_GLOBAL_SCALE_OFFSETS.index(1)
    choices = torch.full(
        (*prefix, block_count),
        fill_value=initial_index,
        dtype=torch.int64,
        device=values.device,
    )
    current = reconstructed[initial_index].clone()
    values_float = values.to(torch.float32)
    projected = (values_float - current) @ projection.T
    order = values_float.reshape(
        -1, block_count, 64
    ).square().sum(dim=(0, 2)).argsort(descending=True).tolist()
    for block in order:
        start = block * 64
        stop = start + 64
        old = current[..., start:stop]
        scores = []
        for candidate in reconstructed:
            new = candidate[..., start:stop]
            candidate_projected = (
                projected + (old - new) @ projection[:, start:stop].T
            )
            local = (
                (values_float[..., start:stop] - new).square()
                * diagonal[start:stop]
            ).sum(dim=-1)
            scores.append(candidate_projected.square().sum(dim=-1) + local)
        best = torch.stack(scores, dim=-1).argmin(dim=-1)
        choices[..., block] = best
        stacked = torch.stack(
            [candidate[..., start:stop] for candidate in reconstructed], dim=-2
        )
        gather_index = best[..., None, None].expand(*best.shape, 1, 64)
        new = torch.gather(stacked, -2, gather_index).squeeze(-2)
        projected.add_((old - new) @ projection[:, start:stop].T)
        current[..., start:stop] = new
    return _select_linear_candidate_params(candidates, choices, values.ndim - 1)


def _select_linear_candidate_params(
    candidates: list[dict[str, torch.Tensor]],
    choices: torch.Tensor,
    prefix_dims: int,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name in candidates[0]:
        selected = torch.zeros_like(candidates[0][name])
        tail_dims = selected.ndim - prefix_dims - 1
        for index, candidate in enumerate(candidates):
            mask = (choices == index).reshape(
                *choices.shape,
                *([1] * tail_dims),
            )
            selected = torch.where(mask, candidate[name], selected)
        result[name] = selected
    return result


def _quantize_linear_global_output(
    values: torch.Tensor,
    quantized_diagonal: torch.Tensor,
    quantized_projection: torch.Tensor,
    max_sweeps: int = _LINEAR_GLOBAL_JOINT_SWEEPS,
    multi_sweep_max_rows: int = _LINEAR_GLOBAL_MULTI_SWEEP_MAX_ROWS,
    scale_offsets: tuple[int, ...] = _LINEAR_GLOBAL_SCALE_OFFSETS,
    task_rounding: bool = False,
) -> dict[str, torch.Tensor]:
    """Approximately minimize ||(A-Q(A))Q(W).T|| sequentially."""
    specifications = [(offset, 0.5) for offset in scale_offsets]
    if task_rounding:
        specifications = [
            specification
            for specification in specifications
            if specification[0] != 6
        ] + [(1, 0.625)]
    candidates = [
        _quantize_linear_values(
            values,
            quantized_diagonal.reshape(-1, 8, 2, 4),
            offsets=(offset,),
            rounding_threshold=threshold,
        )
        for offset, threshold in specifications
    ]
    reconstructed = [_dequantize_linear_params(params) for params in candidates]
    block_count = int(values.shape[-1]) // 64
    prefix = values.shape[:-1]
    initial_index = specifications.index((1, 0.5))
    choices = torch.full(
        (*prefix, block_count),
        fill_value=initial_index,
        dtype=torch.int64,
        device=values.device,
    )
    current = reconstructed[initial_index].clone()
    values_float = values.to(torch.float32)
    projected = (values_float - current) @ quantized_projection.T
    order = values_float.reshape(
        -1, block_count, 64
    ).square().sum(dim=(0, 2)).argsort(descending=True).tolist()
    flat_values = values_float.reshape(-1, values_float.shape[-1])
    flat_current = current.reshape_as(flat_values)
    flat_projected = projected.reshape(-1, projected.shape[-1])
    flat_choices = choices.reshape(-1, block_count)
    flat_candidates = [
        candidate.reshape_as(flat_values) for candidate in reconstructed
    ]
    sweep_count = (
        max_sweeps
        if flat_values.shape[0] <= multi_sweep_max_rows
        else 1
    )
    stacked_candidates = (
        torch.stack(flat_candidates, dim=1)
        if sweep_count > 1 and flat_values.shape[-1] <= 4096
        else None
    )
    for _ in range(sweep_count):
        for block in order:
            start = block * 64
            stop = start + 64
            if stacked_candidates is None:
                stacked = torch.stack(
                    [
                        candidate[:, start:stop]
                        for candidate in flat_candidates
                    ],
                    dim=1,
                )
            else:
                stacked = stacked_candidates[:, :, start:stop]
            projection = quantized_projection[:, start:stop].T
            for row_start in range(0, flat_values.shape[0], _LINEAR_GLOBAL_ROW_CHUNK):
                row_stop = min(
                    row_start + _LINEAR_GLOBAL_ROW_CHUNK,
                    flat_values.shape[0],
                )
                old = flat_current[row_start:row_stop, start:stop]
                new = stacked[row_start:row_stop]
                candidate_projected = flat_projected[
                    row_start:row_stop, None, :
                ] + torch.matmul(old[:, None, :] - new, projection)
                local = (
                    (flat_values[row_start:row_stop, None, start:stop] - new)
                    .square()
                    * quantized_diagonal[start:stop]
                ).sum(dim=-1)
                scores = candidate_projected.square().sum(dim=-1) + local
                best = scores.argmin(dim=-1)
                flat_choices[row_start:row_stop, block] = best
                gather_value = best[:, None, None].expand(-1, 1, 64)
                flat_current[row_start:row_stop, start:stop] = torch.gather(
                    new, 1, gather_value
                ).squeeze(1)
                gather_projection = best[:, None, None].expand(
                    -1, 1, candidate_projected.shape[-1]
                )
                flat_projected[row_start:row_stop] = torch.gather(
                    candidate_projected, 1, gather_projection
                ).squeeze(1)
    return _select_linear_candidate_params(candidates, choices, values.ndim - 1)


def _select_global_activation_policy(
    parts: list[torch.Tensor],
    transformed_weight: torch.Tensor,
    weight_params: dict[str, torch.Tensor],
    activation_state: dict[str, Any],
    force_global: bool,
) -> tuple[str, torch.Tensor, torch.Tensor, float]:
    """Choose baseline, Q(W), or a Q(W)-derived diagonal trust region."""
    samples = [
        _sample_profile_rows(part)[:_LINEAR_GLOBAL_SELECT_ROWS_PER_SHAPE]
        for part in parts
    ]
    sample = torch.cat(samples)
    baseline = _dequantize_linear_params(_quantize_linear_values(
        sample,
        activation_state["importance"],
        offsets=activation_state["activation_offsets"],
        projection=activation_state["projection"],
        projection_strength=activation_state["projection_strength"],
    ))
    base_diagonal = activation_state["global_qw_diagonal"]
    projection = activation_state["global_qw_projection"]
    global_qw = _dequantize_linear_params(_quantize_linear_global_output(
        sample,
        base_diagonal,
        projection,
        int(activation_state["global_joint_sweeps"]),
        int(activation_state["global_multi_sweep_max_rows"]),
        task_rounding=(int(activation_state.get("linear_shape", 0)) == 0),
    ))
    mean_metric_diagonal = (
        base_diagonal + projection.square().sum(dim=0)
    ).mean()
    ridge_diagonal = (
        base_diagonal
        + _LINEAR_GLOBAL_RIDGE_RATIO * mean_metric_diagonal
    )
    global_ridge = _dequantize_linear_params(_quantize_linear_global_output(
        sample,
        ridge_diagonal,
        projection,
        int(activation_state["global_joint_sweeps"]),
        int(activation_state["global_multi_sweep_max_rows"]),
        task_rounding=(int(activation_state.get("linear_shape", 0)) == 0),
    ))
    quantized_weight = _dequantize_linear_params(weight_params)
    target = sample @ transformed_weight.T
    candidates = (baseline, global_qw, global_ridge)
    aggregate_errors = [
        (target - candidate @ quantized_weight.T).square().mean()
        for candidate in candidates
    ]
    layout_errors = []
    start = 0
    for sampled in samples:
        stop = start + int(sampled.shape[0])
        layout_errors.append(torch.stack([
            (
                target[start:stop]
                - candidate[start:stop] @ quantized_weight.T
            ).square().mean()
            for candidate in candidates
        ]))
        start = stop
    layout_errors = torch.stack(layout_errors).T

    selected = 0
    required_gain = 0.0 if force_global else _LINEAR_GLOBAL_SELECT_MIN_GAIN
    for candidate_index in (1, 2):
        if not bool(
            aggregate_errors[candidate_index]
            < (1.0 - required_gain) * aggregate_errors[0]
        ):
            continue
        ratios = layout_errors[candidate_index] / layout_errors[0].clamp_min(
            torch.finfo(torch.float32).tiny
        )
        wins = int((ratios < 1.0).sum())
        if (
            wins >= _LINEAR_GLOBAL_JOINT_MIN_WIN_LAYOUTS
            and bool(ratios.max() <= _LINEAR_GLOBAL_JOINT_MAX_LAYOUT_REGRESSION)
        ):
            if selected == 0 or bool(
                aggregate_errors[candidate_index] < aggregate_errors[selected]
            ):
                selected = candidate_index
    return (
        ("baseline", "qw_global", "qw_global")[selected],
        aggregate_errors[selected],
        layout_errors[selected],
        _LINEAR_GLOBAL_RIDGE_RATIO if selected == 2 else 0.0,
    )


def _apply_selected_global_ridge(
    activation_state: dict[str, Any],
    ridge_ratio: float,
) -> None:
    """Freeze the calibration-selected Q(W)-only metric in online state."""
    activation_state["global_metric_ridge"] = float(ridge_ratio)
    if ridge_ratio <= 0.0:
        return
    diagonal = activation_state["global_qw_diagonal"]
    projection = activation_state["global_qw_projection"]
    mean_metric_diagonal = (
        diagonal + projection.square().sum(dim=0)
    ).mean()
    activation_state["global_qw_diagonal"] = (
        diagonal + ridge_ratio * mean_metric_diagonal
    )


def _install_global_output_state(
    activation_state: dict[str, Any],
    weight_params: dict[str, torch.Tensor],
) -> None:
    quantized_weight = _dequantize_linear_params(weight_params)
    quantized_diagonal, quantized_projection = _global_output_cost_state(
        quantized_weight
    )
    activation_state.update({
        "global_qw_diagonal": quantized_diagonal,
        "global_qw_projection": quantized_projection,
    })


def _prune_global_output_state(
    activation_state: dict[str, Any],
    policy: str,
) -> None:
    activation_state["global_policy"] = policy
    if policy == "baseline":
        activation_state.pop("global_qw_diagonal")
        activation_state.pop("global_qw_projection")


def _activation_output_cost_state(
    transformed_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate each 64-channel block of W.T@W as diagonal + low rank."""
    num_blocks = int(transformed_weight.shape[-1]) // 64
    weight_blocks = transformed_weight.to(torch.float32).reshape(
        -1, num_blocks, 64
    )
    generator = torch.Generator(device=transformed_weight.device).manual_seed(20260820)
    projection = torch.randint(
        0,
        2,
        (num_blocks, 64, _LINEAR_OUTPUT_RANK),
        generator=generator,
        dtype=torch.float32,
        device=transformed_weight.device,
    ).mul_(2.0).sub_(1.0)
    projection = torch.linalg.qr(projection, mode="reduced")[0]
    for _ in range(_LINEAR_POWER_ITERS):
        output_projection = torch.einsum(
            "obd,bdr->obr", weight_blocks, projection
        )
        projection = torch.einsum(
            "obd,obr->bdr", weight_blocks, output_projection
        )
        projection = torch.linalg.qr(projection, mode="reduced")[0]

    output_projection = torch.einsum(
        "obd,bdr->obr", weight_blocks, projection
    )
    strength = output_projection.square().sum(dim=0).transpose(0, 1)
    diagonal = weight_blocks.square().sum(dim=0)
    diagonal_residual = (
        diagonal
        - (
            strength.transpose(0, 1).unsqueeze(1) * projection.square()
        ).sum(dim=-1)
    ).clamp_min(0.0)
    return (
        diagonal_residual.reshape(num_blocks, 8, 2, 4),
        projection.permute(2, 0, 1).reshape(
            _LINEAR_OUTPUT_RANK, num_blocks, 8, 2, 4
        ),
        strength,
    )


def _residual_feedback_weight(
    quantized_activation: torch.Tensor,
    weight: torch.Tensor,
    weight_importance: torch.Tensor,
    weight_projection: torch.Tensor,
    weight_projection_strength: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Propagate each quantized block's output residual into nearby blocks."""
    rows = int(quantized_activation.shape[0])
    if rows > _LINEAR_FEEDBACK_ROWS:
        indices = torch.linspace(
            0,
            rows - 1,
            _LINEAR_FEEDBACK_ROWS,
            device=quantized_activation.device,
        ).round().to(torch.int64)
        activation = quantized_activation.index_select(0, indices)
    else:
        activation = quantized_activation

    channels = int(weight.shape[-1])
    block_count = channels // 64
    weight_offsets = (
        _LINEAR_WEIGHT_SATURATION_OFFSETS
        if (
            channels > int(weight.shape[0])
            and channels <= 2 * int(weight.shape[0])
        )
        else _LINEAR_WEIGHT_SCALE_OFFSETS
    )
    work = weight.to(torch.float32).clone()
    if _LINEAR_FEEDBACK_ENERGY_ORDER:
        block_energy = quantized_activation.reshape(
            -1, block_count, 64
        ).square().sum(dim=(0, 2))
        order = block_energy.argsort(
            descending=_LINEAR_FEEDBACK_DESCENDING
        ).tolist()
    else:
        order = list(range(block_count))
    pieces: dict[str, list[torch.Tensor | None]] = {}
    tiny = torch.finfo(torch.float32).tiny
    for position, block in enumerate(order):
        block_indices = torch.arange(
            block * 64,
            (block + 1) * 64,
            device=work.device,
        )
        work_block = work.index_select(1, block_indices)
        params = _quantize_linear_values(
            work_block,
            weight_importance[block:block + 1],
            offsets=weight_offsets,
            projection=weight_projection[:, block:block + 1],
            projection_strength=weight_projection_strength[:, block:block + 1],
        )
        quantized_block = _dequantize_linear_params(params)
        for name, tensor in params.items():
            pieces.setdefault(name, [None] * block_count)[block] = tensor

        future_blocks = order[
            position + 1:position + 1 + _LINEAR_FEEDBACK_WINDOW
        ]
        if not future_blocks:
            continue
        future_indices = torch.cat([
            torch.arange(
                future * 64,
                (future + 1) * 64,
                device=work.device,
            )
            for future in future_blocks
        ])
        block_activation = activation.index_select(1, block_indices)
        future_activation = activation.index_select(1, future_indices)
        output_residual = block_activation @ (
            work_block - quantized_block
        ).T
        kernel = future_activation @ future_activation.T
        ridge = _LINEAR_FEEDBACK_RIDGE * kernel.diag().mean().clamp_min(tiny)
        identity = torch.eye(
            kernel.shape[0],
            dtype=kernel.dtype,
            device=kernel.device,
        )
        dual = torch.linalg.solve(kernel + ridge * identity, output_residual)
        correction = (future_activation.T @ dual).T
        future_weight = work.index_select(1, future_indices)
        future_weight.add_(
            correction,
            alpha=_LINEAR_FEEDBACK_STEP,
        )
        work.index_copy_(1, future_indices, future_weight)
    return {
        name: torch.cat(
            [tensor for tensor in tensors if tensor is not None],
            dim=-4,
        )
        for name, tensors in pieces.items()
    }


def _feedback_quantize_weight(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_importance: torch.Tensor,
    activation_projection: torch.Tensor,
    projection_strength: torch.Tensor,
    activation_offsets: Iterable[int],
    weight_importance: torch.Tensor,
    weight_projection: torch.Tensor,
    weight_projection_strength: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Fix Q(A), then build one bounded residual-feedback Q(W)."""
    activation_params = _quantize_linear_values(
        activation,
        activation_importance,
        offsets=activation_offsets,
        projection=activation_projection,
        projection_strength=projection_strength,
    )
    quantized_activation = _dequantize_linear_params(activation_params)
    return _residual_feedback_weight(
        quantized_activation,
        weight,
        weight_importance,
        weight_projection,
        weight_projection_strength,
    )


def _sample_profile_rows(values: torch.Tensor) -> torch.Tensor:
    """Bound profile-search rows while covering the full sequence layout."""
    if values.shape[0] <= _LINEAR_PROFILE_ROWS_PER_SHAPE:
        return values
    indices = torch.linspace(
        0,
        values.shape[0] - 1,
        _LINEAR_PROFILE_ROWS_PER_SHAPE,
        device=values.device,
    ).round().to(torch.int64)
    return values[indices]


def _activation_reconstruction_score(
    parts: list[torch.Tensor],
    offsets: Iterable[int],
) -> torch.Tensor:
    """Equal-shape Q(A) reconstruction score; intentionally has no W input."""
    errors = []
    for part in parts:
        sample = _sample_profile_rows(part)
        params = _quantize_linear_values(sample, offsets=offsets)
        reconstructed = _dequantize_linear_params(params)
        error = (sample - reconstructed).square().mean()
        energy = sample.square().mean().clamp_min(torch.finfo(torch.float32).tiny)
        errors.append(error / energy)
    return torch.stack(errors).mean()


def _select_activation_offsets(
    parts: list[torch.Tensor],
) -> tuple[int, ...]:
    """Choose a synthetic-profile candidate using legal A-only calibration."""
    compact_score = _activation_reconstruction_score(
        parts, _LINEAR_SCALE_OFFSETS
    )
    wide_score = _activation_reconstruction_score(
        parts, _LINEAR_WIDE_SCALE_OFFSETS
    )
    if wide_score < (1.0 - _LINEAR_PROFILE_MIN_GAIN) * compact_score:
        return _LINEAR_WIDE_SCALE_OFFSETS
    return _LINEAR_SCALE_OFFSETS


def _rms(values: torch.Tensor) -> torch.Tensor:
    return (
        values.to(torch.float32)
        .reshape(-1, values.shape[-1])
        .square()
        .mean(0)
        .sqrt()
    )


def _linear_multiplier(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    tiny = torch.finfo(torch.float32).tiny
    multiplier = (right.clamp_min(tiny) / left.clamp_min(tiny)).pow(
        _LINEAR_BALANCE_EXPONENT
    )
    multiplier = multiplier.clamp(1.0 / 16.0, 16.0)
    return torch.where(
        (left > 0) & (right > 0), multiplier, torch.ones_like(multiplier)
    )


def _balanced_linear_permutation(
    activation: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Spread high-range channels evenly across contiguous HiF4 groups."""
    sensitivity = _rms(activation) * _rms(weight)
    ordered = torch.argsort(sensitivity)
    group_count = int(activation.shape[-1]) // 64
    lanes = ordered.reshape(64, group_count).clone()
    lanes[1::2] = lanes[1::2].flip(1)
    positions = torch.arange(64, device=ordered.device)
    bit_reversed = sum(
        ((positions >> bit) & 1) << (5 - bit) for bit in range(6)
    )
    return lanes.transpose(0, 1).index_select(1, bit_reversed).reshape(-1)


def _hadamard(values: torch.Tensor, block: int) -> torch.Tensor:
    result = values.to(torch.float32).reshape(
        -1, values.shape[-1] // block, block
    ).clone()
    stride = 1
    while stride < block:
        pairs = result.reshape(
            *result.shape[:-1], block // (2 * stride), 2, stride
        )
        left = pairs[..., 0, :].clone()
        right = pairs[..., 1, :].clone()
        pairs[..., 0, :] = left + right
        pairs[..., 1, :] = left - right
        stride *= 2
    return (result / block**0.5).reshape_as(values)


def _linear_sign_candidates() -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260819)
    random = torch.randint(0, 2, (7, 64), generator=generator).to(torch.float32)
    return torch.cat((torch.ones(1, 64), random.mul_(2).sub_(1)))


def _peak_ratio(values: torch.Tensor) -> torch.Tensor:
    peak = values.abs().amax(-1).square().sum(0)
    energy = values.square().sum(dim=(0, 2))
    return peak / energy.clamp_min(torch.finfo(torch.float32).tiny)


def _select_linear_signs(
    activation: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    group_count = activation.shape[-1] // 64
    a = activation.reshape(-1, group_count, 64)
    w = weight.reshape(-1, group_count, 64)
    candidates = _linear_sign_candidates()
    scores = [
        _peak_ratio(_hadamard(a * signs, 64))
        + _peak_ratio(_hadamard(w * signs, 64))
        for signs in candidates
    ]
    return candidates[torch.stack(scores).argmin(0)].reshape(-1)


def _linear_transform(values: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
    permutation = state.get("permutation")
    if permutation is not None:
        values = values.index_select(-1, permutation)
    return _hadamard(
        values.to(torch.float32) * state["coefficient"],
        int(state["hadamard"]),
    )


# =============================================================================
# Linear: accelerated output-aware calibration
# =============================================================================

@torch.inference_mode()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    weight = dequantize_nvfp4(weight_quant, weight_scale).to(torch.float32)
    activation_parts = [
        dequantize_nvfp4(quant, scale).to(torch.float32)
        for quant, scale in calib_activation_list
    ]
    activation = torch.cat(activation_parts)
    permutation = None
    if int(weight.shape[-1]) <= 2 * int(weight.shape[0]):
        permutation = _balanced_linear_permutation(activation, weight)
        activation = activation.index_select(-1, permutation)
        weight = weight.index_select(-1, permutation)
        activation_parts = list(torch.split(
            activation,
            [int(part.shape[0]) for part in activation_parts],
        ))
    multiplier = _linear_multiplier(_rms(activation), _rms(weight))
    transformed_activation = activation * multiplier
    transformed_weight = weight / multiplier
    signs = _select_linear_signs(transformed_activation, transformed_weight)
    coefficient = multiplier * signs
    transformed_activation = _hadamard(
        transformed_activation * signs, _LINEAR_HADAMARD
    )
    transformed_weight = _hadamard(transformed_weight * signs, _LINEAR_HADAMARD)
    transformed_parts = list(torch.split(
        transformed_activation,
        [int(part.shape[0]) for part in activation_parts],
    ))
    activation_offsets = _select_activation_offsets(transformed_parts)
    weight_importance = transformed_activation.square().mean(0)
    (
        activation_importance,
        activation_projection,
        projection_strength,
    ) = _activation_output_cost_state(transformed_weight)
    force_global_output = (
        int(transformed_weight.shape[-1])
        >= _LINEAR_GLOBAL_MIN_COMPRESSION * int(transformed_weight.shape[0])
    )
    (
        weight_importance,
        weight_projection,
        weight_projection_strength,
    ) = _activation_output_cost_state(transformed_activation)
    activation_state = {
        "version": 120,
        "coefficient": coefficient,
        "hadamard": _LINEAR_HADAMARD,
        "importance": activation_importance,
        "projection": activation_projection,
        "projection_strength": projection_strength,
        "activation_offsets": activation_offsets,
        "global_joint_sweeps": (
            2
            if (
                transformed_weight.shape[0] < transformed_weight.shape[1]
                and not force_global_output
            )
            else _LINEAR_GLOBAL_JOINT_SWEEPS
        ),
        "global_multi_sweep_max_rows": (
            1024
            if transformed_weight.shape[0] > transformed_weight.shape[1]
            else _LINEAR_GLOBAL_MULTI_SWEEP_MAX_ROWS
        ),
        "linear_shape": (
            -1
            if transformed_weight.shape[0] < transformed_weight.shape[1]
            else (1 if transformed_weight.shape[0] > transformed_weight.shape[1] else 0)
        ),
    }
    if permutation is not None:
        activation_state["permutation"] = permutation
    weight_params = _feedback_quantize_weight(
        transformed_activation,
        transformed_weight,
        activation_importance,
        activation_projection,
        projection_strength,
        activation_offsets,
        weight_importance,
        weight_projection,
        weight_projection_strength,
    )
    quantized_weight = _dequantize_linear_params(weight_params)
    (
        deployed_importance,
        deployed_projection,
        deployed_projection_strength,
    ) = _activation_output_cost_state(quantized_weight)
    activation_state.update({
        "importance": deployed_importance,
        "projection": deployed_projection,
        "projection_strength": deployed_projection_strength,
    })
    _install_global_output_state(
        activation_state, weight_params
    )
    global_policy, aggregate_error, layout_errors, global_ridge = (
        _select_global_activation_policy(
            transformed_parts,
            transformed_weight,
            weight_params,
            activation_state,
            force_global_output,
        )
    )
    _apply_selected_global_ridge(activation_state, global_ridge)
    if not force_global_output:
        _prune_global_output_state(activation_state, global_policy)
        return {"weight_params": weight_params, "activation_state": activation_state}

    feedback_rows = min(_LINEAR_FEEDBACK_ROWS, int(transformed_activation.shape[0]))
    feedback_indices = torch.linspace(
        0,
        int(transformed_activation.shape[0]) - 1,
        feedback_rows,
        device=transformed_activation.device,
    ).round().to(torch.int64)
    feedback_values = transformed_activation.index_select(0, feedback_indices)
    alternating_activation = _dequantize_linear_params(
        _quantize_linear_global_output(
            feedback_values,
            activation_state["global_qw_diagonal"],
            activation_state["global_qw_projection"],
            int(activation_state["global_joint_sweeps"]),
            int(activation_state["global_multi_sweep_max_rows"]),
            task_rounding=(int(activation_state.get("linear_shape", 0)) == 0),
        )
    )
    alternating_weight_params = _residual_feedback_weight(
        alternating_activation,
        transformed_weight,
        weight_importance,
        weight_projection,
        weight_projection_strength,
    )
    alternating_state = activation_state.copy()
    alternating_quantized_weight = _dequantize_linear_params(
        alternating_weight_params
    )
    (
        alternating_importance,
        alternating_projection,
        alternating_projection_strength,
    ) = _activation_output_cost_state(alternating_quantized_weight)
    alternating_state.update({
        "importance": alternating_importance,
        "projection": alternating_projection,
        "projection_strength": alternating_projection_strength,
    })
    _install_global_output_state(
        alternating_state, alternating_weight_params
    )
    (
        alternating_policy,
        alternating_error,
        alternating_layout_errors,
        alternating_ridge,
    ) = _select_global_activation_policy(
        transformed_parts,
        transformed_weight,
        alternating_weight_params,
        alternating_state,
        force_global_output,
    )
    _apply_selected_global_ridge(alternating_state, alternating_ridge)
    alternating_ratios = alternating_layout_errors / layout_errors.clamp_min(
        torch.finfo(torch.float32).tiny
    )
    if (
        bool(
            alternating_error
            < (1.0 - _LINEAR_ALTERNATING_MIN_GAIN) * aggregate_error
        )
        and int((alternating_ratios < 1.0).sum())
        >= _LINEAR_ALTERNATING_MIN_WIN_LAYOUTS
        and bool(
            alternating_ratios.max()
            <= _LINEAR_ALTERNATING_MAX_LAYOUT_REGRESSION
        )
    ):
        weight_params = alternating_weight_params
        activation_state = alternating_state
        global_policy = alternating_policy
    _prune_global_output_state(activation_state, global_policy)
    return {"weight_params": weight_params, "activation_state": activation_state}


@torch.inference_mode()
def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    values = dequantize_nvfp4(activation_quant, activation_scale)
    transformed = _linear_transform(values, activation_state)
    if activation_state.get("global_policy") == "qw_global":
        row_count = transformed.numel() // transformed.shape[-1]
        shape = int(activation_state.get("linear_shape", 0))
        replace_offset5 = (
            (shape == 0 and row_count == 512)
            or (shape == 1 and row_count in (1, 128, 1024))
        )
        scale_offsets = (
            (-1, 0, 1, 2, 3, 4, 6, 7)
            if replace_offset5
            else _LINEAR_GLOBAL_SCALE_OFFSETS
        )
        return _quantize_linear_global_output(
            transformed,
            activation_state["global_qw_diagonal"],
            activation_state["global_qw_projection"],
            int(activation_state["global_joint_sweeps"]),
            int(activation_state["global_multi_sweep_max_rows"]),
            scale_offsets,
            task_rounding=(shape == 0),
        )
    return _quantize_linear_values(
        transformed,
        activation_state["importance"],
        offsets=activation_state["activation_offsets"],
        projection=activation_state["projection"],
        projection_strength=activation_state["projection_strength"],
    )


# =============================================================================
# Attention
# =============================================================================

def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """Build reciprocal per-KV-head Q/K balance from calibration moments."""
    q_values = torch.cat([
        dequantize_nvfp4(*sample["q"]).to(torch.float32)
        for sample in calib_qkv_list
    ])
    k_values = torch.cat([
        dequantize_nvfp4(*sample["k"]).to(torch.float32)
        for sample in calib_qkv_list
    ])
    replicas = q_num_heads // kv_num_heads
    q_energy = q_values.reshape(-1, q_num_heads, head_dim).square().mean(0)
    grouped_q_energy = q_energy.reshape(kv_num_heads, replicas, head_dim)
    q_rms = grouped_q_energy.sqrt().mean(1)
    k_rms = k_values.reshape(-1, kv_num_heads, head_dim).square().mean(0).sqrt()
    tiny = torch.finfo(torch.float32).tiny
    balance = (k_rms.clamp_min(tiny) / q_rms.clamp_min(tiny)).sqrt()
    balance = balance.clamp(1.0 / 16.0, 16.0)
    balance = torch.where(
        (q_rms > 0) & (k_rms > 0), balance, torch.ones_like(balance)
    )
    hadamard = 1
    global_rms = k_values.square().mean().sqrt()
    if bool(global_rms > 0):
        near_zero = (
            k_values.abs() < global_rms * _ATTENTION_NEAR_ZERO_RATIO
        ).to(torch.float32).mean()
        if bool(near_zero <= _ATTENTION_SPARSE_THRESHOLD):
            hadamard = head_dim
    return {
        "q_state": {
            "coefficient": balance.repeat_interleave(replicas, dim=0),
            "hadamard": hadamard,
        },
        "k_state": {
            "coefficient": balance.reciprocal(),
            "hadamard": hadamard,
        },
        "v_state": None,
    }


def _quantize_attention_values(
    values: torch.Tensor,
    offsets: tuple[int, ...],
    energy_penalty: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Select legal scales with an optional per-block energy drift cost."""
    candidates = [
        _quantize_linear_values(values, offsets=(offset,))
        for offset in offsets
    ]
    if len(candidates) == 1:
        return candidates[0]
    channels = int(values.shape[-1])
    target = values.to(torch.float32).reshape(
        *values.shape[:-1], channels // 64, 64
    )
    target_energy = target.square().sum(-1).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    scores = []
    for params in candidates:
        reconstructed = _dequantize_linear_params(params).reshape_as(target)
        residual_cost = (target - reconstructed).square().sum(-1)
        if energy_penalty:
            energy_drift = target_energy - reconstructed.square().sum(-1)
            residual_cost = residual_cost + (
                energy_penalty * energy_drift.square() / target_energy
            )
        scores.append(residual_cost)
    best = torch.stack(scores).argmin(0)
    result: dict[str, torch.Tensor] = {}
    for name in candidates[0]:
        stacked = torch.stack([candidate[name] for candidate in candidates])
        tail = stacked.shape[1 + best.ndim:]
        index = best.reshape((1,) + best.shape + (1,) * len(tail))
        index = index.expand((1,) + best.shape + tail)
        result[name] = torch.gather(stacked, 0, index).squeeze(0)
    return result


def _transform_attention_qk(
    values: torch.Tensor,
    heads: int,
    head_dim: int,
    state: dict[str, Any],
) -> torch.Tensor:
    shaped = values.reshape(-1, heads, head_dim)
    coefficient = state["coefficient"].to(values.device, values.dtype)
    shaped = shaped * coefficient.unsqueeze(0)
    return _hadamard(shaped, int(state["hadamard"])).reshape_as(values)


def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    value = dequantize_nvfp4(q_quant, q_scale).to(torch.float32)
    value = _transform_attention_qk(value, q_num_heads, head_dim, q_state)
    return _quantize_attention_values(
        value,
        _ATTENTION_QK_OFFSETS,
        _ATTENTION_ENERGY_PENALTY,
    )


def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    value = dequantize_nvfp4(k_quant, k_scale).to(torch.float32)
    value = _transform_attention_qk(value, kv_num_heads, head_dim, k_state)
    return _quantize_attention_values(
        value,
        _ATTENTION_QK_OFFSETS,
        _ATTENTION_ENERGY_PENALTY,
    )


def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    value = dequantize_nvfp4(v_quant, v_scale).to(torch.float32)
    return _quantize_attention_values(value, _ATTENTION_V_OFFSETS)


# =============================================================================
# Online-proven v10 Attention override
# =============================================================================

_ATTN_V10_OFFSETS = (-1, 0, 1, 2, 3)


def _attn_v10_e6m2_ceil_index(scale: torch.Tensor) -> torch.Tensor:
    safe = scale.float().clamp(min=E6M2_MIN, max=E6M2_MAX)
    exponent = torch.floor(torch.log2(safe))
    normalized = safe * torch.pow(2.0, -exponent)
    mantissa = torch.ceil((normalized - 1.0) * 4.0)
    carry = mantissa >= 4.0
    exponent = exponent + carry.to(exponent.dtype)
    mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa)
    return (((exponent + 48.0) * 4.0 + mantissa).to(torch.int64)).clamp_(0, 254)


def _attn_v10_e6m2_from_index(index: torch.Tensor) -> torch.Tensor:
    index = index.clamp(0, 254)
    exponent = torch.div(index, 4, rounding_mode="floor").float() - 48.0
    mantissa = torch.remainder(index, 4).float()
    value = torch.pow(2.0, exponent) * (1.0 + mantissa * 0.25)
    return value.clamp_(E6M2_MIN, E6M2_MAX)


def _attn_v10_scaled_error(
    grouped: torch.Tensor,
    scale: torch.Tensor,
    multiplier: float,
    rounding_threshold: float,
) -> torch.Tensor:
    local_scale = scale[..., None, None, None] * multiplier
    scaled = grouped.abs() / local_scale * 4.0
    if rounding_threshold < 0.0:
        mant = torch.round(scaled) * 0.25
    else:
        mant = (
            torch.floor(scaled)
            + (scaled - torch.floor(scaled) >= rounding_threshold)
        ) * 0.25
    reconstructed = torch.sign(grouped) * mant.clamp_(0.0, 1.75) * local_scale
    return (reconstructed - grouped).square().sum(dim=-1)


def _attn_v10_tree_error(
    grouped: torch.Tensor,
    scale: torch.Tensor,
    rounding_threshold: float,
) -> torch.Tensor:
    err1 = _attn_v10_scaled_error(grouped, scale, 1.0, rounding_threshold)
    err2 = _attn_v10_scaled_error(grouped, scale, 2.0, rounding_threshold)
    err4 = _attn_v10_scaled_error(grouped, scale, 4.0, rounding_threshold)
    level2_one = torch.minimum(err1, err2).sum(dim=-1)
    level2_two = torch.minimum(err2, err4).sum(dim=-1)
    return torch.minimum(level2_one, level2_two).sum(dim=-1)


def _attn_v10_materialize_tree(
    grouped: torch.Tensor,
    scale: torch.Tensor,
    rounding_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    err1 = _attn_v10_scaled_error(grouped, scale, 1.0, rounding_threshold)
    err2 = _attn_v10_scaled_error(grouped, scale, 2.0, rounding_threshold)
    err4 = _attn_v10_scaled_error(grouped, scale, 4.0, rounding_threshold)
    level2_one = torch.minimum(err1, err2).sum(dim=-1)
    level2_two = torch.minimum(err2, err4).sum(dim=-1)
    use_level2_two = level2_two < level2_one
    level2 = torch.where(use_level2_two, 2.0, 1.0)
    use_level3 = torch.where(
        use_level2_two.unsqueeze(-1),
        err4 < err2,
        err2 < err1,
    )
    level3 = torch.where(use_level3, 2.0, 1.0)
    total_scale = (
        scale[..., None, None, None]
        * level2[..., None, None]
        * level3[..., None]
    )
    scaled = grouped.abs() / total_scale * 4.0
    if rounding_threshold < 0.0:
        mant = (torch.round(scaled) * 0.25).clamp_(0.0, 1.75)
    else:
        mant = (
            (
                torch.floor(scaled)
                + (scaled - torch.floor(scaled) >= rounding_threshold)
            )
            * 0.25
        ).clamp_(0.0, 1.75)
    sign = torch.where(mant == 0.0, 0.0, torch.sign(grouped))
    return level2, level3, sign, mant


def _attn_v10_quantize(
    values: torch.Tensor,
    offsets: tuple[int, ...] = _ATTN_V10_OFFSETS,
    rounding_threshold: float = -1.0,
) -> dict[str, torch.Tensor]:
    values = values.float()
    shape = tuple(int(dim) for dim in values.shape)
    channels = shape[-1]
    grouped = values.unflatten(-1, (-1, 64)).reshape(
        *shape[:-1], channels // 64, 8, 2, 4
    )
    peak = grouped.abs().amax(dim=(-1, -2, -3))
    base_index = _attn_v10_e6m2_ceil_index(peak / 7.0)
    best_error = torch.full_like(peak, torch.inf)
    best_index = base_index
    for offset in offsets:
        candidate_index = (base_index + int(offset)).clamp(0, 254)
        candidate_scale = _attn_v10_e6m2_from_index(candidate_index)
        candidate_error = _attn_v10_tree_error(
            grouped, candidate_scale, rounding_threshold
        )
        improve = candidate_error < best_error
        best_error = torch.where(improve, candidate_error, best_error)
        best_index = torch.where(improve, candidate_index, best_index)
    scale = _attn_v10_e6m2_from_index(best_index)
    level2, level3, sign, mant = _attn_v10_materialize_tree(
        grouped, scale, rounding_threshold
    )
    return {
        "scale_factor": scale.reshape(*shape[:-1], channels // 64, 1, 1, 1),
        "scale_lv2": level2.reshape(*shape[:-1], channels // 64, 8, 1, 1),
        "scale_lv3": level3.reshape(*shape[:-1], channels // 64, 8, 2, 1),
        "sign": sign.reshape(*shape[:-1], channels // 64, 8, 2, 4),
        "mant": mant.reshape(*shape[:-1], channels // 64, 8, 2, 4),
    }


def _attn_v10_dequantize(params: dict[str, torch.Tensor]) -> torch.Tensor:
    values = (
        params["sign"]
        * params["mant"]
        * params["scale_lv2"]
        * params["scale_lv3"]
        * params["scale_factor"]
    )
    return values.reshape(*params["sign"].shape[:-4], -1)


def _attn_v10_fwht(values: torch.Tensor, block_size: int) -> torch.Tensor:
    channels = int(values.shape[-1])
    prefix = tuple(int(value) for value in values.shape[:-1])
    blocks = values.reshape(*prefix, channels // block_size, block_size)
    width = 1
    while width < block_size:
        paired = blocks.reshape(
            *prefix,
            channels // block_size,
            block_size // (2 * width),
            2,
            width,
        )
        left, right = paired[..., 0, :], paired[..., 1, :]
        blocks = torch.cat((left + right, left - right), dim=-1).reshape(
            *prefix, channels // block_size, block_size
        )
        width *= 2
    return blocks.reshape(*prefix, channels) * (block_size ** -0.5)


def _attn_v10_sign(
    channels: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if seed == 0:
        return torch.ones(channels, dtype=torch.float32, device=device)
    index = torch.arange(channels, dtype=torch.int64, device=device)
    bit = torch.bitwise_and((index * 1103515245 + seed * 12345) >> 16, 1)
    return bit.float().mul_(2.0).sub_(1.0).contiguous()


def _attn_v10_block_sizes(channels: int) -> tuple[int, ...]:
    largest = 1
    while largest * 2 <= min(channels, 2048) and channels % (largest * 2) == 0:
        largest *= 2
    values = {64, largest}
    if channels % 256 == 0:
        values.add(256)
    return tuple(sorted(value for value in values if channels % value == 0))


def _attn_v10_transform(
    values: torch.Tensor,
    permutation: torch.Tensor,
    sign: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    return _attn_v10_fwht(
        values.index_select(-1, permutation.to(values.device, torch.int64))
        * sign.to(values.device, torch.float32),
        block_size,
    )


def _attn_v10_head_transform(
    values: torch.Tensor,
    heads: int,
    head_dim: int,
    permutation: torch.Tensor,
    sign: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    rows = int(values.shape[0])
    return _attn_v10_transform(
        values.reshape(rows, heads, head_dim),
        permutation,
        sign,
        block_size,
    ).reshape(rows, heads * head_dim)


def _attn_v10_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    rows = int(q.shape[0])
    q = q.reshape(rows, q_heads, head_dim).transpose(0, 1)
    k = k.reshape(rows, kv_heads, head_dim).transpose(0, 1).repeat_interleave(
        q_heads // kv_heads, 0
    )
    v = v.reshape(rows, kv_heads, head_dim).transpose(0, 1).repeat_interleave(
        q_heads // kv_heads, 0
    )
    return torch.softmax(
        (q @ k.transpose(-1, -2)) / (head_dim ** 0.5), -1
    ) @ v


def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    fit_samples = calib_qkv_list[:-1] if len(calib_qkv_list) > 1 else calib_qkv_list
    valid_samples = (
        calib_qkv_list[-2:] if len(calib_qkv_list) >= 3 else calib_qkv_list[-1:]
    )
    replicas = q_num_heads // kv_num_heads
    fit_q = torch.cat([
        dequantize_nvfp4(*sample["q"]).float() for sample in fit_samples
    ])
    fit_k = torch.cat([
        dequantize_nvfp4(*sample["k"]).float() for sample in fit_samples
    ])
    q_peak = fit_q.abs().reshape(-1, q_num_heads, head_dim).amax((0, 1))
    k_peak = fit_k.abs().reshape(-1, kv_num_heads, head_dim).amax((0, 1))
    identity = torch.arange(head_dim, device=fit_q.device)
    sorted_permutation = torch.argsort(torch.maximum(q_peak, k_peak))
    decoded = [
        (
            dequantize_nvfp4(*sample["q"]).float(),
            dequantize_nvfp4(*sample["k"]).float(),
            dequantize_nvfp4(*sample["v"]).float(),
        )
        for sample in valid_samples
    ]
    references = [
        _attn_v10_attention(q, k, v, q_num_heads, kv_num_heads, head_dim)
        for q, k, v in decoded
    ]
    quantized_v = [
        _attn_v10_dequantize(_attn_v10_quantize(v, offsets=(0,)))
        for _, _, v in decoded
    ]
    best_error = float("inf")
    best = (identity, _attn_v10_sign(head_dim, 0, fit_q.device), 64)
    best_smooth: torch.Tensor | None = None
    for permutation in (identity, sorted_permutation):
        for seed in (0, 1):
            sign = _attn_v10_sign(head_dim, seed, fit_q.device)
            for block_size in _attn_v10_block_sizes(head_dim):
                transformed_q = _attn_v10_head_transform(
                    fit_q, q_num_heads, head_dim, permutation, sign, block_size
                )
                transformed_k = _attn_v10_head_transform(
                    fit_k, kv_num_heads, head_dim, permutation, sign, block_size
                )
                candidate_q_peak = (
                    transformed_q.abs()
                    .reshape(-1, q_num_heads, head_dim)
                    .amax(0)
                    .reshape(kv_num_heads, replicas, head_dim)
                    .amax(1)
                )
                candidate_k_peak = (
                    transformed_k.abs()
                    .reshape(-1, kv_num_heads, head_dim)
                    .amax(0)
                )
                smooth = torch.sqrt(
                    candidate_q_peak.clamp_min(1e-8)
                    / candidate_k_peak.clamp_min(1e-8)
                ).clamp(1 / 16, 16)
                q_smooth = smooth.repeat_interleave(replicas, 0).reshape(-1)
                k_smooth = smooth.reshape(-1)
                error = 0.0
                for index, (q, k, _) in enumerate(decoded):
                    tq = _attn_v10_head_transform(
                        q, q_num_heads, head_dim, permutation, sign, block_size
                    )
                    tk = _attn_v10_head_transform(
                        k, kv_num_heads, head_dim, permutation, sign, block_size
                    )
                    output = _attn_v10_attention(
                        _attn_v10_dequantize(
                            _attn_v10_quantize(tq / q_smooth, offsets=(0,))
                        ),
                        _attn_v10_dequantize(
                            _attn_v10_quantize(tk * k_smooth, offsets=(0,))
                        ),
                        quantized_v[index],
                        q_num_heads,
                        kv_num_heads,
                        head_dim,
                    )
                    error += float((output - references[index]).square().mean())
                if error < best_error:
                    best_error = error
                    best = (permutation, sign, block_size)
                    best_smooth = smooth
    permutation, sign, block_size = best
    assert best_smooth is not None
    q_smooth = best_smooth.repeat_interleave(replicas, 0).reshape(-1)
    k_smooth = best_smooth.reshape(-1)
    fixed_qk = []
    for q, k, _ in decoded:
        tq = _attn_v10_head_transform(
            q, q_num_heads, head_dim, permutation, sign, block_size
        )
        tk = _attn_v10_head_transform(
            k, kv_num_heads, head_dim, permutation, sign, block_size
        )
        fixed_qk.append(
            (
                _attn_v10_dequantize(_attn_v10_quantize(tq / q_smooth)),
                _attn_v10_dequantize(_attn_v10_quantize(tk * k_smooth)),
            )
        )
    best_v_threshold = -1.0
    best_v_error = float("inf")
    for threshold in (-1.0, 0.375, 0.5, 0.625):
        error = 0.0
        for index, (_, _, v) in enumerate(decoded):
            q_quantized, k_quantized = fixed_qk[index]
            v_quantized = _attn_v10_dequantize(
                _attn_v10_quantize(v, rounding_threshold=threshold)
            )
            output = _attn_v10_attention(
                q_quantized,
                k_quantized,
                v_quantized,
                q_num_heads,
                kv_num_heads,
                head_dim,
            )
            error += float((output - references[index]).square().mean())
        if error < best_v_error:
            best_v_error = error
            best_v_threshold = threshold
    common = {
        "permutation": permutation.cpu().to(torch.int32).contiguous(),
        "sign": sign.cpu().to(torch.int8).contiguous(),
        "block_size": int(block_size),
    }
    return {
        "q_state": {
            **common,
            "smooth": best_smooth.repeat_interleave(replicas, 0)
            .reshape(-1).cpu().float().contiguous(),
        },
        "k_state": {
            **common,
            "smooth": best_smooth.reshape(-1).cpu().float().contiguous(),
        },
        "v_state": {"rounding_threshold": float(best_v_threshold)},
    }


def _attn_v10_dynamic(
    quant: torch.Tensor,
    scale: torch.Tensor,
    heads: int,
    head_dim: int,
    state: dict[str, Any],
    inverse: bool,
) -> dict[str, torch.Tensor]:
    values = dequantize_nvfp4(quant, scale).float()
    values = _attn_v10_head_transform(
        values,
        heads,
        head_dim,
        state["permutation"],
        state["sign"],
        int(state["block_size"]),
    )
    smooth = state["smooth"].to(values.device)
    return _attn_v10_quantize(values * smooth if inverse else values / smooth)


def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    return _attn_v10_dynamic(
        q_quant, q_scale, q_num_heads, head_dim, q_state, False
    )


def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    return _attn_v10_dynamic(
        k_quant, k_scale, kv_num_heads, head_dim, k_state, True
    )


def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    values = dequantize_nvfp4(v_quant, v_scale).float()
    return _attn_v10_quantize(
        values,
        rounding_threshold=float(v_state["rounding_threshold"]),
    )
