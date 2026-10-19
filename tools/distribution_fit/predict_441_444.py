"""Predict online scores for v441-v444 from the fitted mixture, anchored on v435.

Correct delta handling: local_delta(v) = [sum over COMMON columns of w*(q_v - q_v435)] / coverage,
where coverage = sum of w over the common columns. This avoids treating unmeasured columns as 0.
online_delta = scale * local_delta, with scale calibrated on the fit's passed edges.
online_total(v) = online_total(v435) + online_delta   (v435 online total = 21304)
attention(v) = online_total(v) - 4450 (user's linear assumption)
"""
import json
from pathlib import Path
D = Path(__file__).resolve().parents[2] / 'docs/distribution-fit'
LINEAR = 4450.0
ANCHOR_V, ANCHOR_TOTAL = 435, 21304.0

fit = json.loads((D / 'amplitude-20pct-vmean-growth.json').read_text())
es = fit['edges']
num = sum(e['predicted_total_delta'] * e['target'] for e in es if e['amplitude_passed'])
den = sum(e['predicted_total_delta'] ** 2 for e in es if e['amplitude_passed'])
scale = num / den
keys = json.loads((D / 'score-keys.json').read_text()); cols = keys['columns']

def q(v, label):
    rnd, atom, case = label.split('/'); test = int(case.split('-')[0][4:])
    p = D / rnd / 'cells' / ('v%s--%s.json' % (v, atom))
    if not p.exists(): return None
    c = json.loads(p.read_text())
    if not c.get('complete'): return None
    for x in c['cases']:
        if x['test'] == test: return x['quality']
    return None

def delta(v):
    tot = 0.0; cov = 0.0; miss = 0
    for l, w in cols.items():
        a = q(ANCHOR_V, l); b = q(v, l)
        if a is None or b is None: miss += 1; continue
        tot += w * (b - a); cov += w
    return ((tot / cov) * 25000.0 if cov > 0 else 0.0), cov, miss

print('anchor v%d = %.0f (linear %.0f assumed => attention %.0f)'
      % (ANCHOR_V, ANCHOR_TOTAL, LINEAR, ANCHOR_TOTAL - LINEAR))
print('calibration scale (fit passed edges) = %.4f' % scale)
print()
print('%-6s %9s %10s %12s %13s %11s %10s' % ('ver', 'cols', 'coverage', 'local_delta', 'online_delta', 'est_total', 'est_attn'))
res = {}
for v in [441, 442, 443, 444]:
    dl, cov, miss = delta(v)
    od = scale * dl
    tot = ANCHOR_TOTAL + od
    res[v] = {'local_delta': dl, 'coverage': cov, 'missing_columns': miss,
              'online_delta': od, 'est_online_total': tot, 'est_attention': tot - LINEAR,
              'complete': miss == 0}
    print('%-6s %6d/63 %9.1f%% %12.1f %13.1f %11.0f %10.0f'
          % ('v%d' % v, 63 - miss, cov * 100, dl, od, tot, tot - LINEAR))
out = {'anchor_version': ANCHOR_V, 'anchor_online_total': ANCHOR_TOTAL, 'linear_assumed': LINEAR,
       'scale_on_fit_passed_edges': scale,
       'method': 'fitted 675-col mixture (amplitude-20pct-vmean-growth.json); '
                 'local_delta over common columns scaled by weight coverage; online_delta = scale*local_delta',
       'predictions': {str(k): v for k, v in res.items()},
       'limits': 'Extrapolation from training-support fit (77/98 amplitude matched, direction 98/98). '
                 'Fit residual RMSE on passed edges ~158; small-magnitude band (<100) ~28. '
                 'v443 has incomplete column coverage.'}
(D / 'score-prediction-441-444.json').write_text(json.dumps(out, indent=2) + '\n')
print('\nsaved', D / 'score-prediction-441-444.json')
