"""Generic evaluator: run one solution version on a public reference attention dataset.

For each group in the dataset:
  - three independent role calibrations from the group's `calib` list
    (call <module>.hif4_calibration_attention(samples, qh, kh, d), keep only the role's state)
  - per test case: deployed = <module>.hif4_dynamic_quantize_<role>(...); baseline = quantize_to_hif4
  - quality = (MSE_STD - MSE_PLAYER)/MSE_STD   with reference = NVFP4-dequantised inputs
Case weight: dataset `fit_cases[].integer_count` when present, else 1 (equal weight).
"""
import argparse, copy, json, sys, time, traceback
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.test_v422 import load
from tools.verify_attention_campaign import equal
from hif4_quant import dequantize_nvfp4, dequantize_hif4, quantize_to_hif4
from evaluate_local import gqa_attention, mse, score
from self_check import validate_attention_calibration_result, validate_hif4_params

@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, help='path to attn.pt')
    p.add_argument('--version', type=int, required=True)
    p.add_argument('--groups', help='comma list of group indices (default all)')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(4); torch.manual_seed(20260910)
    path = Path(a.dataset)
    groups = torch.load(path, weights_only=True)
    assert isinstance(groups, list), 'expected list of groups'
    idx = list(range(len(groups))) if not a.groups else [int(x) for x in a.groups.split(',')]
    record = {'dataset': str(path), 'version': a.version, 'n_groups': len(groups),
              'groups_used': idx, 'complete': False, 'cases': [], 'format_checks': 0}
    started = time.perf_counter()
    try:
        for gi in idx:
            g = groups[gi]
            frozen = copy.deepcopy(g)
            qh, kh, d = g['q_num_heads'], g['kv_num_heads'], g['head_dim']
            states = {}
            for role in 'qkv':
                m = load(a.version, 'calib_%s_%d' % (role, gi))
                samples = copy.deepcopy(g['calib'])
                calibrated = m.hif4_calibration_attention(samples, qh, kh, d)
                assert not validate_attention_calibration_result(calibrated, 'fit')
                states[role + '_state'] = copy.deepcopy(calibrated[role + '_state'])
            # case weights
            counts = None
            fc = g.get('fit_cases')
            if fc:
                counts = [float(c.get('integer_count', 1)) for c in fc]
            for i, s in enumerate(g['test']):
                raw = {r: dequantize_nvfp4(*s[r]).float() for r in 'qkv'}
                deployed = {}; baseline = {}
                for role in 'qkv':
                    m = load(a.version, 'dyn_%s_%d_%d' % (role, gi, i))
                    st = copy.deepcopy(states[role + '_state']); saved = copy.deepcopy(st)
                    qp = getattr(m, 'hif4_dynamic_quantize_' + role)(
                        *s[role], qh if role == 'q' else kh, d, st)
                    assert not validate_hif4_params(qp, raw[role].shape, role)
                    record['format_checks'] += 1
                    assert equal(st, saved)
                    deployed[role] = dequantize_hif4(qp, raw[role].shape)
                    baseline[role] = dequantize_hif4(quantize_to_hif4(raw[role]), raw[role].shape)
                target = gqa_attention(*[raw[r] for r in 'qkv'], qh, kh, d)
                bm = mse(target, gqa_attention(*[baseline[r] for r in 'qkv'], qh, kh, d))
                pm = mse(target, gqa_attention(*[deployed[r] for r in 'qkv'], qh, kh, d))
                v = score(bm, pm)
                assert torch.isfinite(torch.tensor(v))
                w = counts[i] if counts and i < len(counts) else 1.0
                record['cases'].append({'group': gi, 'test': i, 'tokens': len(raw['v']),
                                        'weight': w, 'quality': v})
            assert equal(g, frozen)
        tw = sum(c['weight'] for c in record['cases'])
        record['mean_quality'] = sum(c['quality'] * c['weight'] for c in record['cases']) / tw
        record['unweighted_mean_quality'] = sum(c['quality'] for c in record['cases']) / len(record['cases'])
        record['total_weight'] = tw
        record['complete'] = True
    except Exception:
        record['error'] = traceback.format_exc()
    record['wall_seconds'] = time.perf_counter() - started
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp'); tmp.write_text(json.dumps(record, indent=2) + '\n'); tmp.replace(out)
    print(json.dumps({k: record.get(k) for k in
                      ('dataset', 'version', 'complete', 'mean_quality', 'unweighted_mean_quality', 'wall_seconds')}))
    if not record['complete']:
        print(record.get('error', '')[-1500:]); raise SystemExit(1)

if __name__ == '__main__':
    main()
