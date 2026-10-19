"""Current-K softmax-distillation search over invariant translation gauges."""
import contextlib
import copy
import io
import json
import time

import torch

from evaluate_local import evaluate_attention
from probe_v319_value import ROOT, load_solution
from self_check import validate_attention_calibration_result, validate_hif4_params


TAIL = r'''
DK_DEBUG = {}
_dk_parent_calibration = hif4_calibration_attention
_dk_parent_k = hif4_dynamic_quantize_k


def _dk_queries(samples, parent, qh, kh, d):
    pieces = []
    for sample in samples[:-1][-4:]:
        count = sample['q'][0].shape[0]
        ix = torch.linspace(0, count - 1, min(count, 16)).long()
        raw = dequantize_nvfp4(sample['q'][0][ix], sample['q'][1][ix]).float()
        query = _kt_input(raw, parent['q_state'], qh, d, 'q')
        query = query.reshape(-1, kh, qh // kh, d).permute(1, 0, 2, 3).reshape(kh, -1, d)
        pieces.append(query / d ** .5)
    return torch.cat(pieces, 1).contiguous()


def _dk_cross_entropy(query, target_probability, key):
    logits = query @ key.transpose(0, 1).transpose(1, 2)
    return (torch.logsumexp(logits, -1) - (target_probability * logits).sum(-1)).mean(-1)


def _dk_quant(raw, st, kh, d, active=None):
    parent = _dk_parent_k(raw['quant'], raw['scale'], kh, d, st)
    source = dequantize_nvfp4(raw['quant'], raw['scale']).float()
    if source.shape[0] <= 1:
        return parent
    y = _kt_input(source, st, kh, d, 'k').reshape(-1, kh, d)
    query = st['dk_query'].float()
    target_probability = torch.softmax(query @ y.transpose(0, 1).transpose(1, 2), -1)
    weight = (.95 * target_probability.mean(1) + .05 / y.shape[0]).transpose(0, 1)[:, :, None]
    weighted = (y * weight).sum(0, keepdim=True)
    uniform = y.mean(0, keepdim=True)
    median = y.median(0, keepdim=True).values
    midrange = (y.amin(0, keepdim=True) + y.amax(0, keepdim=True)) * .5
    centers = [torch.zeros_like(uniform), uniform, weighted, median, midrange]
    center = weighted
    for step in range(8):
        shifted = y - center
        decoded = _deq_params(_dyn_table(shifted.reshape(source.shape), None, has_scale=False)).reshape_as(y)
        center = center - ((decoded - shifted) * weight).sum(0, keepdim=True)
        if step in (1, 3, 7):
            centers.append(center)
    losses = []
    for center in centers:
        shifted = y - center
        decoded = _deq_params(_dyn_table(shifted.reshape(source.shape), None, has_scale=False)).reshape_as(y)
        losses.append(_dk_cross_entropy(query, target_probability, decoded))
    choice = torch.stack(losses).argmin(0)
    chosen = torch.gather(torch.cat(centers, 0), 0, choice[None, :, None].expand(1, kh, d))
    plain = {k: v for k, v in st.items() if k not in ('joint_transform', 'ks', 'rot', 'hc_matrix')}
    trial = _kt_fixed_grid((y - chosen).reshape(source.shape).contiguous(), plain, kh, d, 'k')
    parent_value = _deq_params(parent).reshape_as(y)
    trial_value = _deq_params(trial).reshape_as(y)
    old = _dk_cross_entropy(query, target_probability, parent_value)
    new = _dk_cross_entropy(query, target_probability, trial_value)
    take = new < old
    if active is not None:
        take &= active
    mask = take.repeat_interleave(d // 64)
    merged = {name: torch.where(mask.reshape(1, -1, *([1] * (value.ndim - 2))), value, parent[name])
              for name, value in trial.items()}
    return merged


def _dk_guard(samples, parent, qh, kh, d):
    DK_DEBUG.clear()
    DK_DEBUG.update(accepted=0, error=False)
    if len(samples) < 3 or d % 64:
        return parent
    hold = samples[-1]
    rows = hold['k'][0].shape[0]
    if not 4 <= rows <= 4096:
        return parent
    proposed = dict(parent['k_state'], dk_query=_dk_queries(samples, parent, qh, kh, d))
    ix = torch.linspace(0, rows - 1, min(rows, 64)).long()
    raw = [dequantize_nvfp4(hold['q'][0][ix], hold['q'][1][ix]).float()]
    raw.extend(dequantize_nvfp4(*hold[r]).float() for r in 'kv')
    target = _joint_output(*raw, qh, kh, d, ix, query_subset=True)
    query = _joint_quant_queries(hold, parent['q_state'], qh, kh, d, ix)
    value = _deq_params(hif4_dynamic_quantize_v(*hold['v'], kh, d, parent['v_state']))
    old_key = _deq_params(_dk_parent_k(*hold['k'], kh, d, parent['k_state']))
    packed = {'quant': hold['k'][0], 'scale': hold['k'][1]}
    new_key = _deq_params(_dk_quant(packed, proposed, kh, d))
    def cost(key):
        error = (_joint_output(query, key, value, qh, kh, d, ix, query_subset=True) - target).square().mean((-1, -2))
        return torch.stack((error[::2].mean(0), error[1::2].mean(0)))
    old = cost(old_key)
    new = cost(new_key)
    ratio = new / old.clamp_min(1e-30)
    take = (ratio.mean(0) < .997) & (ratio.amax(0) < 1.01)
    DK_DEBUG.update(accepted=int(take.sum()), ratio=ratio.mean(0).tolist(), worst=ratio.amax(0).tolist())
    if not bool(take.any()):
        return parent
    return dict(parent, k_state=dict(proposed, dk_active=take))


def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    parent = _dk_parent_calibration(calib_qkv_list, q_num_heads, kv_num_heads, head_dim)
    try:
        return _dk_guard(calib_qkv_list, parent, q_num_heads, kv_num_heads, head_dim)
    except Exception as exc:
        DK_DEBUG.update(error=True, error_type=type(exc).__name__)
        return parent


def hif4_dynamic_quantize_k(k_quant, k_scale, kv_num_heads, head_dim, k_state):
    if not isinstance(k_state, dict) or 'dk_active' not in k_state or not 4 <= k_quant.shape[0] <= 4096:
        return _dk_parent_k(k_quant, k_scale, kv_num_heads, head_dim, k_state)
    return _dk_quant({'quant': k_quant, 'scale': k_scale}, k_state, kv_num_heads, head_dim, k_state['dk_active'])
'''


def main():
    torch.set_num_threads(4)
    bank = torch.load(ROOT / 'local_fitted_models_attention_feedback_v2/attn.pt', weights_only=False)
    base = load_solution(ROOT / 'solution/solution_v324.py')
    trial = load_solution(ROOT / 'solution/solution_v324.py')
    exec(compile(TAIL, 'dynamic_key_softmax_bank', 'exec'), trial.__dict__)
    rows = []
    for group_index in (0, 1, 2, 5, 14, 19, 25, 35, 47, 49):
        group = bank[group_index]
        dims = [group[k] for k in ('q_num_heads', 'kv_num_heads', 'head_dim')]
        start = time.perf_counter()
        parent = trial._dk_parent_calibration(group['calib'], *dims)
        state = trial._dk_guard(group['calib'], parent, *dims)
        calibration_seconds = time.perf_counter() - start
        assert not validate_attention_calibration_result(state, 'dynamic-key')
        debug = copy.deepcopy(trial.DK_DEBUG)
        base.hif4_calibration_attention = lambda *unused, _state=parent: copy.deepcopy(_state)
        trial.hif4_calibration_attention = lambda *unused, _state=state: copy.deepcopy(_state)
        with contextlib.redirect_stdout(io.StringIO()):
            baseline, _ = evaluate_attention(base, [group])
            scores, _ = evaluate_attention(trial, [group])
        for sample in group['test']:
            params = trial.hif4_dynamic_quantize_k(*sample['k'], dims[1], dims[2], state['k_state'])
            assert not validate_hif4_params(params, sample['k'][0].shape, 'dynamic-key')
        row = dict(group=group_index, baseline=baseline, scores=scores,
                   delta=sum(scores) - sum(baseline), calibration_seconds=calibration_seconds,
                   debug=debug)
        rows.append(row)
        print(row, flush=True)
    report = dict(complete=True, rows=rows, delta=sum(r['delta'] for r in rows),
                  warning='Modeled activations are not online score evidence.')
    (ROOT / 'docs/dynamic-key-softmax-bank-screen.json').write_text(json.dumps(report, indent=2) + '\n')
    print({'delta': report['delta'], 'accepted': sum(r['debug']['accepted'] for r in rows)}, flush=True)


if __name__ == '__main__':
    main()
