"""Bounded, self-only V content discrepancy search on a fixed HiF4 lattice.

The caller owns deployment eligibility. No Q/K, learned kernel, calibration
gate, mutable global state, or random numbers are used here. Two rounds keep
four strictly improving one-code leaders per channel, and enumerate every
other token's -1/0/+1 partner. At most 16 content-discrepant channels are
searched per call. This is a finite shortlist, not a global solve.
"""

import torch


def _v435_buckets(values):
    """Four content regions from the two highest-variance current V channels."""
    directions = values.double().var(0, unbiased=False).topk(2).indices
    features = values[:, directions]
    bits = features > features.median(0).values
    return bits[:, 0].long() + 2 * bits[:, 1].long()


def _v435_statistics(error, buckets):
    """Per-channel exact-double statistics; error is [channels, tokens]."""
    error = error.double()
    index = buckets[None].expand(error.shape[0], -1)
    counts = torch.bincount(buckets, minlength=4).double().clamp_min(1)
    sums = torch.zeros(error.shape[0], 4, dtype=torch.float64)
    sums.scatter_add_(1, index, error)
    total = error.sum(1)
    sse = error.square().sum(1)
    objective = (sums.square() / counts).sum(1)
    objective = objective + (total.square() + sse) / error.shape[1]
    return objective, sse, total.square(), sums, total, counts


def _v435_decode_codes(codes, steps):
    # Every proposal is judged after BF16 rounding, not at an ideal FP64 point.
    return (codes.double() * steps).to(torch.bfloat16).double()


def _v435_search_chunk(target, initial_codes, steps, buckets):
    """Search a small [channels,tokens] chunk while preserving parent caps."""
    codes = initial_codes.clone()
    values = _v435_decode_codes(codes, steps)
    error = values - target
    _, original_sse, original_mean, _, _, _ = _v435_statistics(error, buckets)
    channels, tokens = error.shape
    row = torch.arange(channels)
    index = buckets[None].expand(channels, -1)

    for _ in range(2):
        objective, sse, _, sums, total, counts = _v435_statistics(error, buckets)
        inverse_count = counts.reciprocal()[buckets]
        gradient = sums.gather(1, index) * inverse_count
        gradient = gradient + (total[:, None] + error) / tokens
        diagonal = inverse_count + 2.0 / tokens
        moves, losses = [], []
        for direction in (-1, 1):
            next_codes = codes + direction
            change = _v435_decode_codes(next_codes, steps) - values
            loss = 2 * gradient * change + diagonal * change.square()
            legal = (next_codes >= -7) & (next_codes <= 7) & (change != 0)
            moves.append(change)
            losses.append(torch.where(legal, loss, torch.full_like(loss, float('inf'))))
        flat_loss = torch.cat(losses, 1)
        flat_move = torch.cat(moves, 1)
        leaders = flat_loss.topk(min(4, 2 * tokens), dim=1, largest=False).indices
        best = torch.zeros(channels, dtype=torch.float64)
        best_i = torch.zeros(channels, dtype=torch.long)
        best_j = best_i.clone()
        best_di = best_i.clone()
        best_dj = best_i.clone()

        for rank in range(leaders.shape[1]):
            leader = leaders[:, rank]
            first = leader % tokens
            first_direction = torch.where(leader < tokens, -1, 1)
            first_change = flat_move[row, leader]
            first_loss = flat_loss[row, leader]
            first_error = error[row, first]
            first_sse = 2 * first_error * first_change + first_change.square()
            same_bucket = buckets[first, None] == buckets[None]
            coupling = 1.0 / tokens + same_bucket * inverse_count[None]
            leader_valid = torch.isfinite(first_loss) & (first_loss < 0)
            if not bool(leader_valid.any()):
                continue

            for direction in (-1, 0, 1):
                if direction == 0:
                    second_change = torch.zeros_like(values)
                    second_loss = torch.zeros_like(values)
                    legal = torch.ones_like(codes, dtype=torch.bool)
                else:
                    side = 0 if direction < 0 else 1
                    second_change, second_loss = moves[side], losses[side]
                    legal = torch.isfinite(second_loss)
                delta = first_loss[:, None] + second_loss
                delta = delta + 2 * first_change[:, None] * second_change * coupling
                trial_sse = sse[:, None] + first_sse[:, None]
                trial_sse = trial_sse + 2 * error * second_change + second_change.square()
                trial_mean = (total[:, None] + first_change[:, None] + second_change).square()
                accept = legal & leader_valid[:, None]
                accept = accept & (trial_sse <= original_sse[:, None])
                accept = accept & (trial_mean <= original_mean[:, None])
                accept[row, first] = False  # the two coordinates must differ
                candidate = torch.where(accept, delta, torch.full_like(delta, float('inf')))
                loss, second = candidate.min(1)
                take = loss < best
                best = torch.where(take, loss, best)
                best_i = torch.where(take, first, best_i)
                best_j = torch.where(take, second, best_j)
                best_di = torch.where(take, first_direction, best_di)
                best_dj = torch.where(take, direction, best_dj)

        trial_codes = codes.clone()
        trial_codes[row, best_i] += best_di
        trial_codes[row, best_j] += best_dj
        trial_values = _v435_decode_codes(trial_codes, steps)
        trial_error = trial_values - target
        trial_cost, trial_sse, trial_mean, _, _, _ = _v435_statistics(trial_error, buckets)
        # Recompute the complete objective and original-parent caps. There is
        # no epsilon relaxation; exact equality is permitted only for the caps.
        take = (trial_cost < objective) & (trial_sse <= original_sse)
        take = take & (trial_mean <= original_mean) & torch.isfinite(trial_cost)
        if not bool(take.any()):
            break  # identical state would repeat the same deterministic search
        codes = torch.where(take[:, None], trial_codes, codes)
        values = torch.where(take[:, None], trial_values, values)
        error = values - target
    return codes


def _v435_decode_params(params, shape):
    value = params['sign'] * params['mant']
    value = value * params['scale_lv2'] * params['scale_lv3'] * params['scale_factor']
    return value.reshape(shape).to(torch.bfloat16).double()


def _v435_refine(x, params, enabled, kh, d):
    """Return legal parameters with self-content risk down and both caps held.

    The original scale hierarchy is fixed. Disabled heads and all bypasses
    preserve original fields. The Pareto property concerns V error statistics,
    not an unknown true Attention probability or the full Q/K/V output loss.
    """
    if (x.device.type != 'cpu' or x.ndim != 2 or d < 2 or d % 64
            or kh < 1 or x.shape[1] != kh * d or not 2 <= x.shape[0] <= 4096
            or x.numel() > 2 ** 21 or enabled is None or enabled.numel() != kh
            or enabled.device.type != 'cpu' or not bool(enabled.any())):
        return params
    if not bool(torch.isfinite(x).all()):
        return params
    if any(value.device.type != 'cpu' for value in params.values()):
        return params
    shape = x.shape
    original_codes = (params['sign'].float() * params['mant'].float() * 4).reshape(shape)
    codes = original_codes.clone()
    unit = params['scale_factor'].double() * params['scale_lv2'].double()
    unit = unit * params['scale_lv3'].double()
    steps = unit.expand_as(params['mant']).reshape(shape) * 0.25
    if not bool(torch.isfinite(steps).all()) or not bool((steps > 0).all()):
        return params
    target = x.double()
    original_values = _v435_decode_params(params, shape)
    regions = {}
    priority = torch.full((kh * d,), -float('inf'), dtype=torch.float64)
    for head in range(kh):
        if not bool(enabled[head]):
            continue
        begin = head * d
        bucket = _v435_buckets(x[:, begin:begin + d])
        regions[head] = bucket
        stat = _v435_statistics(
            (original_values[:, begin:begin + d] - target[:, begin:begin + d]).T, bucket)
        # Component of the bucket-constant residual orthogonal to the global
        # constant. Spend the fixed 16-channel budget where this content-only
        # discrepancy is largest; no learned probability or test ID is used.
        score = (stat[3].square() / stat[5]).sum(1) - stat[4].square() / shape[0]
        priority[begin:begin + d] = score
    rank = priority.topk(min(16, int(enabled.sum()) * d)).indices
    selected = rank[torch.isfinite(priority[rank]) & (priority[rank] > 0)]
    if selected.numel() == 0:
        return params
    for head, bucket in regions.items():
        columns = selected[selected // d == head]
        if columns.numel() == 0:
            continue
        codes[:, columns] = _v435_search_chunk(
            target[:, columns].T.contiguous(), original_codes[:, columns].T.contiguous(),
            steps[:, columns].T.contiguous(), bucket).T

    packed_codes = codes.reshape(params['mant'].shape)
    changed = packed_codes != original_codes.reshape(params['mant'].shape)
    result = dict(params)
    result['sign'] = torch.where(changed, packed_codes.sign().to(params['sign'].dtype), params['sign'])
    result['mant'] = torch.where(changed, (packed_codes.abs() * 0.25).to(params['mant'].dtype), params['mant'])
    deployed = _v435_decode_params(result, shape)
    keep = torch.zeros(kh * d, dtype=torch.bool)
    for head, bucket in regions.items():
        columns = selected[selected // d == head]
        if columns.numel() == 0:
            continue
        old = _v435_statistics((original_values[:, columns] - target[:, columns]).T, bucket)
        new = _v435_statistics((deployed[:, columns] - target[:, columns]).T, bucket)
        keep[columns] = (new[0] < old[0]) & (new[1] <= old[1]) & (new[2] <= old[2])
        keep[columns] &= torch.isfinite(new[0])
    mask = keep.reshape(1, kh * d // 64, 8, 2, 4)
    for key in ('sign', 'mant'):
        result[key] = torch.where(mask, result[key], params[key]).contiguous()
    return result
