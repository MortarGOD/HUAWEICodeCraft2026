"""Test whether role-local calibration can select native V over full v435."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.test_v422 import load
from tools.check_role_decoy_calibration import role_samples
from tools.verify_attention_campaign import equal
from hif4_quant import dequantize_nvfp4, dequantize_hif4
from self_check import validate_hif4_params, validate_attention_calibration_result


@torch.inference_mode()
def evaluate(m, sample, states, qh, kh, d):
    raw = {r: dequantize_nvfp4(*sample[r]).float() for r in 'qkv'}
    par = {r: getattr(m, 'hif4_dynamic_quantize_' + r)(
        *sample[r], qh if r == 'q' else kh, d, states[r + '_state']) for r in 'qkv'}
    par['native'] = m._dyn_table(raw['v'], None, False)
    dec = {}
    for r, p in par.items():
        shape = raw['v'].shape if r == 'native' else raw[r].shape
        assert not validate_hif4_params(p, shape, r)
        dec[r] = dequantize_hif4(p, shape)
    t = len(raw['v'])
    ix = torch.linspace(0, t - 1, min(32, t)).long()
    prob = m._vr_probs(raw['q'][ix], raw['k'], qh, kh, d).reshape(kh, -1, t)
    pq = m._vr_probs(dec['q'][ix], dec['k'], qh, kh, d).reshape(kh, -1, t)
    target = prob @ raw['v'].reshape(t, kh, d).transpose(0, 1)
    losses = []
    for r in ('v', 'native'):
        out = pq @ dec[r].reshape(t, kh, d).transpose(0, 1)
        losses.append((out - target).square().mean((1, 2)))
    changed = (dec['v'] != dec['native']).reshape(t, kh, d).sum((0, 2))
    return torch.stack(losses), changed


@torch.inference_mode()
def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--groups', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--candidate', type=int)
    args = ap.parse_args()
    torch.set_num_threads(4)
    path = ROOT / 'docs/papers/hif4-public-references/test/datasets' / args.dataset / 'attn.pt'
    bank = torch.load(path, weights_only=True)
    report = {'completed': False, 'dataset': args.dataset,
              'data_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'parent_sha256': hashlib.sha256((ROOT / 'solution/solution_v435.py').read_bytes()).hexdigest(),
              'groups': [], 'warning': 'Decoy hypotheses are not judge reconstruction. Short captured traces not independently recaptured. No online score prediction.'}
    candidate = load(args.candidate, 'actual_route') if args.candidate else None
    if candidate is not None:
        report['candidate_sha256'] = hashlib.sha256((ROOT / f'solution/solution_v{args.candidate}.py').read_bytes()).hexdigest()
    dest = ROOT / args.output
    for mode in ('paired', 'row_permute', 'block_resample'):
        for gi in map(int, args.groups.split(',')):
            g = bank[gi]
            qh, kh, d = [g[k] for k in ('q_num_heads', 'kv_num_heads', 'head_dim')]
            states = {}
            if mode == 'paired':
                m = load(435, 'route_paired')
                vcal = g['calib']
                vcontext = m.hif4_calibration_attention(vcal, qh, kh, d)
                states = copy.deepcopy(vcontext)
            else:
                for r in 'qkv':
                    m = load(435, 'route_' + r)
                    cal = role_samples(g['calib'], r, mode, 829 + ord(r))
                    result = m.hif4_calibration_attention(cal, qh, kh, d)
                    assert not validate_attention_calibration_result(result, 'context')
                    states[r + '_state'] = copy.deepcopy(result[r + '_state'])
                    if r == 'v':
                        vcal, vcontext = cal, result
            frozen = copy.deepcopy(states)
            context_frozen = copy.deepcopy(vcontext)
            errors = torch.stack([evaluate(m, s, vcontext, qh, kh, d)[0] for s in vcal])
            ratios = errors[:, 1] / errors[:, 0].clamp_min(1e-30)
            choices = {'strict': (ratios < .99).all(0), 'average': ratios.mean(0) < .99}
            rec = {'mode': mode, 'group': gi, 'calibration_ratios': ratios.tolist(),
                   'selected': {k: v.tolist() for k, v in choices.items()}, 'tests': []}
            if candidate is not None:
                start = time.perf_counter()
                routed = candidate._v440_fit(vcal, vcontext, qh, kh, d)
                rec['extra_fit_seconds'] = time.perf_counter() - start
                routed_snapshot = copy.deepcopy(routed)
                rec['actual_heads'] = routed.get('v440_native_heads', torch.zeros(kh, dtype=torch.bool)).tolist()
                valid = [i for i, s in enumerate(vcal) if candidate._v440_eligible(s['v'][0].shape[0], kh, d)][:3]
                expected = ratios[valid].mean(0) < .99 if len(valid) >= 2 else torch.zeros(kh, dtype=torch.bool)
                assert rec['actual_heads'] == expected.tolist()
            for s in g['test']:
                losses, changed = evaluate(m, s, states, qh, kh, d)
                den = losses[0].mean().clamp_min(1e-30)
                values = {k: float(torch.where(mask, losses[1], losses[0]).mean() / den)
                          for k, mask in choices.items()}
                values.update(native=float(losses[1].mean() / den),
                              oracle=float(losses.min(0).values.mean() / den))
                rec['tests'].append({'mse_by_head': losses.tolist(), 'ratios': values,
                                     'native_changed_by_head': changed.tolist()})
                if candidate is not None:
                    start = time.perf_counter()
                    actual = candidate.hif4_dynamic_quantize_v(*s['v'], kh, d, routed)
                    rec['tests'][-1]['actual_v_seconds'] = time.perf_counter() - start
                    assert not validate_hif4_params(actual, s['v'][0].shape, 'candidate')
                    parent_par = m.hif4_dynamic_quantize_v(*s['v'], kh, d, states['v_state'])
                    native_par = m._dyn_table(m.dequantize_nvfp4(*s['v']).float(), None, False)
                    t = s['v'][0].shape[0]
                    a = dequantize_hif4(actual, s['v'][0].shape).reshape(t, kh, d)
                    old = dequantize_hif4(parent_par, s['v'][0].shape).reshape(t, kh, d)
                    native = dequantize_hif4(native_par, s['v'][0].shape).reshape(t, kh, d)
                    mask = torch.tensor(rec['actual_heads']) if candidate._v440_eligible(t, kh, d) else torch.zeros(kh, dtype=torch.bool)
                    assert torch.equal(a, torch.where(mask[None, :, None], native, old))
                    values['actual'] = float(torch.where(mask, losses[1], losses[0]).mean() / den)
                    rec['tests'][-1]['actual_changed'] = int((a != old).sum())
                    fresh = load(args.candidate, 'actual_fresh')
                    assert equal(actual, fresh.hif4_dynamic_quantize_v(*s['v'], kh, d, copy.deepcopy(routed)))
                    assert equal(routed, routed_snapshot)
            assert equal(frozen, states) and equal(context_frozen, vcontext)
            report['groups'].append(rec)
            dest.write_text(json.dumps(report, indent=2) + '\n')
            print({'mode': mode, 'group': gi, 'heads': {k: int(v.sum()) for k, v in choices.items()},
                   'ratios': {k: sum(t['ratios'][k] for t in rec['tests']) / len(rec['tests'])
                              for k in ('strict', 'average', 'native', 'oracle')}}, flush=True)
    report['summary'] = {}
    for mode in ('paired', 'row_permute', 'block_resample'):
        groups = [g for g in report['groups'] if g['mode'] == mode]
        tests = [t for g in groups for t in g['tests']]
        keys = ('strict', 'average', 'native', 'oracle', 'actual') if candidate is not None else ('strict', 'average', 'native', 'oracle')
        report['summary'][mode] = {k: sum(t['ratios'][k] for t in tests) / len(tests) for k in keys}
    report['completed'] = True
    dest.write_text(json.dumps(report, indent=2) + '\n')
    print(report['summary'])


if __name__ == '__main__':
    run()
