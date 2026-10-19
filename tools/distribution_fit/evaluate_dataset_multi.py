"""Evaluate MANY solution versions on one public reference attention dataset in a single process.

Loads the dataset once and evaluates each requested version over the selected groups.
Writes a matrix: {version: {per-case qualities, weighted mean}}.
Case weight: dataset fit_cases[].integer_count when present, else 1.
"""
import argparse, copy, json, os, sys, time, traceback
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
    p.add_argument('--dataset', required=True)
    p.add_argument('--versions', required=True, help='comma list')
    p.add_argument('--groups', help='comma list of group idx (default all)')
    p.add_argument('--weights', help='optional JSON with counts keyed by fit_cases label')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(int(os.environ.get('EVAL_THREADS', '4'))); torch.manual_seed(20260910)
    groups = torch.load(Path(a.dataset), weights_only=True)
    assert isinstance(groups, list)
    gi_list = list(range(len(groups))) if not a.groups else [int(x) for x in a.groups.split(',')]
    versions = [int(x) for x in a.versions.split(',')]
    ext_counts = None
    if a.weights:
        wj = json.loads(Path(a.weights).read_text())
        ext_counts = wj.get('counts', wj)
    # precompute NVFP4 reference + baseline once per (group,test) — version independent
    prep = {}
    for gi in gi_list:
        g = groups[gi]
        qh, kh, d = g['q_num_heads'], g['kv_num_heads'], g['head_dim']
        fc = g.get('fit_cases')
        if fc and ext_counts is not None:
            counts = [float(ext_counts.get(c.get('label'), 0.0)) for c in fc]
        elif fc:
            counts = [float(c.get('integer_count', 1)) for c in fc]
        else:
            counts = None
        cases = []
        for i, s in enumerate(g['test']):
            raw = {r: dequantize_nvfp4(*s[r]).float() for r in 'qkv'}
            target = gqa_attention(*[raw[r] for r in 'qkv'], qh, kh, d)
            baseline = {r: dequantize_hif4(quantize_to_hif4(raw[r]), raw[r].shape) for r in 'qkv'}
            bm = mse(target, gqa_attention(*[baseline[r] for r in 'qkv'], qh, kh, d))
            w = counts[i] if counts and i < len(counts) else 1.0
            cases.append({'test': i, 'tokens': len(raw['v']), 'weight': w, 'raw': raw,
                          'target': target, 'baseline_mse': bm})
        prep[gi] = {'qh': qh, 'kh': kh, 'd': d, 'cases': cases}
    out = {'dataset': a.dataset, 'versions': versions, 'groups_used': gi_list,
           'n_groups_total': len(groups), 'results': {}}
    for v in versions:
        t0 = time.perf_counter(); rec = {'cases': [], 'complete': False}
        try:
            for gi in gi_list:
                g = groups[gi]; frozen = copy.deepcopy(g)
                qh, kh, d = prep[gi]['qh'], prep[gi]['kh'], prep[gi]['d']
                states = {}
                for role in 'qkv':
                    m = load(v, 'calib_%s_%d' % (role, gi))
                    cal = m.hif4_calibration_attention(copy.deepcopy(g['calib']), qh, kh, d)
                    assert not validate_attention_calibration_result(cal, 'fit')
                    states[role + '_state'] = copy.deepcopy(cal[role + '_state'])
                for c in prep[gi]['cases']:
                    raw, target = c['raw'], c['target']
                    deployed = {}
                    for role in 'qkv':
                        m = load(v, 'dyn_%s_%d_%d' % (role, gi, c['test']))
                        st = copy.deepcopy(states[role + '_state']); saved = copy.deepcopy(st)
                        qp = getattr(m, 'hif4_dynamic_quantize_' + role)(
                            *g['test'][c['test']][role], qh if role == 'q' else kh, d, st)
                        assert not validate_hif4_params(qp, raw[role].shape, role)
                        assert equal(st, saved)
                        deployed[role] = dequantize_hif4(qp, raw[role].shape)
                    pm = mse(target, gqa_attention(*[deployed[r] for r in 'qkv'], qh, kh, d))
                    rec['cases'].append({'group': gi, 'test': c['test'], 'tokens': c['tokens'],
                                         'weight': c['weight'], 'quality': score(c['baseline_mse'], pm)})
                assert equal(g, frozen)
            tw = sum(c['weight'] for c in rec['cases'])
            rec['mean_quality'] = sum(c['quality'] * c['weight'] for c in rec['cases']) / tw
            rec['complete'] = True
        except Exception:
            rec['error'] = traceback.format_exc()[-1200:]
        rec['wall_seconds'] = time.perf_counter() - t0
        out['results'][str(v)] = rec
        print('v%d complete=%s mq=%.6f %.1fs' % (v, rec['complete'], rec.get('mean_quality', float('nan')), rec['wall_seconds']), flush=True)
    dest = Path(a.output); dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.tmp'); tmp.write_text(json.dumps(out, indent=2) + '\n'); tmp.replace(dest)
    print('WROTE', dest)

if __name__ == '__main__':
    main()
