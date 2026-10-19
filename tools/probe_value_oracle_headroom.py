"""Offline-only sampled-output oracle; never a deployable independent V interface.

Uses test Q/K to diagnose lost curvature/cross-error information. Modeled
activations are not real model-forward captures. No online score extrapolation.
"""
import argparse
import json
import time
import torch
from probe_v319_value import ROOT, load_solution


def search(p, target, base, step, rounds):
    value = base.clone()
    residual = p @ value - target
    diagonal = p.square().sum(1).clamp_min(1e-30)
    initial = residual.square().sum((1, 2))
    history = []
    for iteration in range(rounds):
        gradient = p.transpose(1, 2) @ residual
        curvature = diagonal[:, :, None]
        candidate = ((value-gradient/curvature)/step).round().clamp(-7, 7)*step
        delta = candidate-value
        cost = 2*gradient*delta+curvature*delta.square()
        best, index = cost.min(1)
        move = delta.gather(1, index[:, None]).squeeze(1)*(best < 0)
        value.scatter_add_(1, index[:, None], move[:, None])
        selected = p.gather(2, index[:, None].expand(-1, p.shape[1], -1))
        residual = residual+selected*move[:, None]
        if iteration+1 in (1, 8, 32):
            exact = p @ value-target
            loss = exact.square().sum((1, 2))
            assert bool((loss <= initial+1e-5*initial.clamp_min(1e-8)).all())
            assert torch.allclose(residual, exact, atol=2e-5, rtol=2e-4)
            history.append(dict(rounds=iteration+1, loss=loss.tolist()))
    return value, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups', default='0,1,5,14,19,49')
    parser.add_argument('--queries', type=int, default=32)
    parser.add_argument('--random-split', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    bank = torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt', weights_only=False)
    m = load_solution(ROOT/'solution/solution_v324.py')
    rows = []
    for gi in map(int, args.groups.split(',')):
        start = time.perf_counter()
        group = bank[gi]
        qh, kh, d = [group[k] for k in ('q_num_heads', 'kv_num_heads', 'head_dim')]
        state = m.hif4_calibration_attention(group['calib'], qh, kh, d)
        sample = group['test'][-1]
        t = sample['v'][0].shape[0]
        ix = torch.linspace(0, t-1, min(args.queries, t)).long()
        q = m._joint_quant_queries(sample, state['q_state'], qh, kh, d, ix)
        k = m._deq_params(m.hif4_dynamic_quantize_k(*sample['k'], kh, d, state['k_state']))
        rawq = m.dequantize_nvfp4(sample['q'][0][ix], sample['q'][1][ix]).float()
        rawk = m.dequantize_nvfp4(*sample['k']).float()
        rawv = m.dequantize_nvfp4(*sample['v']).float().reshape(t, kh, d).transpose(0, 1)
        p = m._vr_probs(q, k, qh, kh, d).reshape(kh, -1, t)
        truep = m._vr_probs(rawq, rawk, qh, kh, d).reshape(kh, -1, t)
        params = m.hif4_dynamic_quantize_v(*sample['v'], kh, d, state['v_state'])
        base = m._deq_params(params).reshape(t, kh, d).transpose(0, 1)
        step = (m._params_unit_flat(params)*.25).reshape(t, kh, d).transpose(0, 1)
        # Hold out half of the sampled queries: avoid calling training fit generalization.
        target = truep @ rawv
        if args.random_split:
            order = torch.randperm(p.shape[1], generator=torch.Generator().manual_seed(350+gi))
            p, target = p[:, order], target[:, order]
        row = dict(group=gi, tokens=t, baseline=float((p@base-target).square().sum()), modes={})
        for name, fit_target in [('curvature_only', p@rawv), ('full_output', target)]:
            out, history = search(p[:, ::2], fit_target[:, ::2], base, step, 32)
            row['modes'][name] = dict(history=history,
                final_output_ratio=float((p@out-target).square().sum()/(p@base-target).square().sum()),
                heldout_ratio=float((p[:, 1::2]@out-target[:, 1::2]).square().sum()/
                                    (p[:, 1::2]@base-target[:, 1::2]).square().sum()))
        row['seconds'] = time.perf_counter()-start
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = dict(complete=True, deployable=False, queries=args.queries, random_split=args.random_split,
                  data='checkpoint-derived modeled activations', rows=rows)
    suffix = '-random' if args.random_split else ''
    (ROOT/f'docs/value-oracle-headroom-q{args.queries}{suffix}.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
