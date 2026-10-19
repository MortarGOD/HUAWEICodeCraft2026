"""Local scoring system: map the fitted mixture score to online attention score.

S_local(v) = 25000 * sum_k w_k * quality_k(v)   over the nonzero-weight (round/atom/test) columns.
online_attention(v) = online_total(v) - LINEAR_ONLINE (=4450, user assumption).
Calibrate the linear map online_attention ~ a * S_local + b on the 102 known versions,
then predict new versions once their columns are measured.
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; D = ROOT / 'docs/distribution-fit'
LINEAR_ONLINE = 4450.0

def load_cols():
    d = json.loads((D / 'amplitude-20pct-vmean-growth.json').read_text())
    return {k: v for k, v in d['weights'].items() if abs(v) > 1e-6}

def quality(version, label):
    rnd, atom, case = label.split('/'); test = int(case.split('-')[0][4:])
    p = D / rnd / 'cells' / ('v%s--%s.json' % (version, atom))
    if not p.exists(): return None
    c = json.loads(p.read_text())
    if not c.get('complete'): return None
    for x in c['cases']:
        if x['test'] == test: return x['quality']
    return None

def s_local(version, cols):
    s = 0.0; miss = 0
    for label, w in cols.items():
        q = quality(version, label)
        if q is None: miss += 1; continue
        s += w * q
    return s * 25000.0, miss

def main():
    cols = load_cols()
    t = json.loads((D / 'targets-20pct.json').read_text())
    rows = []
    for r in t['versions']:
        v = r['version']
        sl, miss = s_local(v, cols)
        if miss: print('WARN missing', miss, 'cols for v', v)
        att = float(r['score']) - LINEAR_ONLINE
        rows.append((v, sl, att, float(r['score'])))
    # regression online_attention = a*S_local + b
    X = np.array([r[1] for r in rows]); Y = np.array([r[2] for r in rows])
    A = np.vstack([X, np.ones_like(X)]).T
    (a, b), res, *_ = np.linalg.lstsq(A, Y, rcond=None)
    pred = a * X + b
    err = pred - Y
    print('calibration on %d versions: online_att = %.6f * S_local + %.1f' % (len(rows), a, b))
    print('  correlation:', round(float(np.corrcoef(X, Y)[0, 1]), 4))
    print('  residual rmse: %.1f  max|err|: %.1f' % (float(np.sqrt((err ** 2).mean())), float(np.abs(err).max())))
    print('  S_local range: %.1f .. %.1f' % (X.min(), X.max()))
    print('  online_att range: %.1f .. %.1f' % (Y.min(), Y.max()))
    out = {'linear_online': LINEAR_ONLINE, 'slope': float(a), 'intercept': float(b),
           'corr': float(np.corrcoef(X, Y)[0, 1]), 'n_versions': len(rows),
           'columns': len(cols),
           'calibration': [{'version': r[0], 's_local': r[1], 'online_attention': r[2], 'online_total': r[3]} for r in rows]}
    (D / 'local-score-calibration.json').write_text(json.dumps(out, indent=2) + '\n')
    # show biggest deviations
    idx = np.argsort(-np.abs(err))[:8]
    print('  largest calibration residuals:')
    for i in idx:
        print('    v%d: S_local=%.1f online_att=%.0f pred=%.0f err=%.0f' % (rows[i][0], X[i], Y[i], pred[i], err[i]))

if __name__ == '__main__':
    main()
