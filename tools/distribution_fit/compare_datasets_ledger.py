"""Score public reference datasets against the authoritative public online ledger.

Ground truth is `tests/benchmarks/commit-history-alignment/public-score-ledger-061e7e6.json`
from the public reference checkout: 204 versions with a public measured online score, of which
113 are `valid_score` inside the single comparable epoch `current_50_linear_250_attention`
(277..440). Scores are comparable ONLY within that epoch.

Two tests are run per dataset:

  LEVEL    online attention(v) = online_score(v) - LINEAR, compared with
           local(v) = mean_quality(v) * 25000, across every evaluated version.
           This is the strong test: it needs the dataset's absolute difficulty to match, not
           merely its sensitivity.
  DELTA    for consecutive evaluated versions inside the epoch, compare
           local(Delta) with online(Delta) - direction, correlation, one global scale,
           and amplitude inside tolerance after that scale.
  HELD OUT v441..444 (user-reported online totals, absent from the ledger) are predicted from
           the v435 anchor using the dataset's own scale.
"""
import json, sys
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parents[2] / 'docs/distribution-fit'
EV = Path(sys.argv[1]) if len(sys.argv) > 1 else (D / 'pub-dataset-eval')
PUB = (Path(__file__).resolve().parents[2] /
       'docs/papers/hif4-public-references/test/tests/benchmarks/commit-history-alignment')
LEDGER = PUB / 'public-score-ledger-061e7e6.json'
EPOCH = 'current_50_linear_250_attention'
LINEAR = 4450.0
ONLINE_441_444 = {441: 21308, 442: 21305, 443: 21304, 444: 21304}
ANCHOR = 435


def load_ledger():
    d = json.loads(LEDGER.read_text())
    scores = {}
    for e in d['entries']:
        if (e['epoch'] == EPOCH and e['result_kind'] == 'valid_score'
                and e['numeric_comparison_allowed'] and e['online_score'] is not None):
            scores[int(e['version'])] = float(e['online_score'])
    return dict(sorted(scores.items()))


def metrics(L, T, thr):
    L = np.asarray(L, float); T = np.asarray(T, float)
    denom = float((L * L).sum())
    scale = float((L * T).sum() / denom) if denom > 0 else float('nan')
    corr = (float(np.corrcoef(L, T)[0, 1]) if len(L) > 2 and L.std() > 0 and T.std() > 0
            else float('nan'))
    rho = (float(np.corrcoef(np.argsort(np.argsort(L)), np.argsort(np.argsort(T)))[0, 1])
           if len(L) > 3 and L.std() > 0 and T.std() > 0 else float('nan'))
    dir_ok = int(sum(1 for l, t in zip(L, T) if t == 0 or (l > 0) == (t > 0)))
    amp_raw = int((np.abs(L - T) <= thr).sum())
    amp_scaled = int((np.abs(scale * L - T) <= thr).sum())
    big = np.array([t / l for l, t in zip(L, T) if abs(l) > 1 and abs(t) >= 100])
    disp_big = (float(np.subtract(*np.percentile(big, [75, 25])) / abs(np.median(big)))
                if len(big) > 3 and np.median(big) != 0 else float('nan'))
    return {'n': len(L), 'scale_ls': scale, 'corr': corr, 'rho': rho, 'dir_ok': dir_ok,
            'amp_raw': amp_raw, 'amp_scaled': amp_scaled, 'disp_big': disp_big,
            'mean_abs_err': float(np.abs(L - T).mean()),
            'mean_abs_err_scaled': float(np.abs(scale * L - T).mean())}


def main():
    ledger = load_ledger()
    rows = []
    for f in sorted(EV.glob('*.json')):
        d = json.loads(f.read_text())
        q = {int(k): v['mean_quality'] for k, v in d['results'].items() if v.get('complete')}
        ev = [v for v in ledger if v in q]
        if len(ev) < 5:
            continue
        # ---- LEVEL: absolute attention level per version
        lv = np.array([q[v] * 25000.0 for v in ev])
        on = np.array([ledger[v] - LINEAR for v in ev])
        lvl_corr = (float(np.corrcoef(lv, on)[0, 1]) if lv.std() > 0 and on.std() > 0
                    else float('nan'))
        lvl_resid = float(np.abs(lv - on).mean())
        # best additive+multiplicative fit of local onto online level
        A = np.vstack([lv, np.ones_like(lv)]).T
        coef, *_ = np.linalg.lstsq(A, on, rcond=None)
        lvl_fit_resid = float(np.abs(A @ coef - on).mean())
        # ---- DELTA: consecutive evaluated versions in ledger order
        Ls, Ts, pairs = [], [], []
        for a, b in zip(ev, ev[1:]):
            Ls.append((q[b] - q[a]) * 25000.0)
            Ts.append(ledger[b] - ledger[a])
            pairs.append((a, b, (q[b] - q[a]) * 25000.0, ledger[b] - ledger[a]))
        Ls = np.array(Ls); Ts = np.array(Ts)
        thr = np.maximum(np.abs(Ts) * 0.2, 1.0)
        m = metrics(Ls, Ts, thr)
        # delta correlation restricted to material online moves
        mask = np.abs(Ts) >= 100
        m_big = metrics(Ls[mask], Ts[mask], thr[mask]) if mask.sum() > 3 else None
        # ---- HELD OUT
        anchor_local = q.get(ANCHOR)
        held = {c: ((q[c] - anchor_local) * 25000.0
                    if (c in q and anchor_local is not None) else None)
                for c in ONLINE_441_444}
        if anchor_local is not None:
            pred = {c: (ledger[ANCHOR] + m['scale_ls'] * held[c]) if held[c] is not None else None
                    for c in ONLINE_441_444}
            perr = {c: (abs(pred[c] - ONLINE_441_444[c]) if pred[c] is not None else None)
                    for c in ONLINE_441_444}
        else:
            pred, perr = {}, {}
        rows.append({'dataset': f.stem, 'n_versions_matched': len(ev),
                     'versions': ev,
                     'level_corr': lvl_corr,
                     'level_mean_abs_err': lvl_resid,
                     'level_oracle_affine_resid': lvl_fit_resid,
                     'level_err_at_v435': (abs(anchor_local * 25000.0 - (ledger[ANCHOR] - LINEAR))
                                           if anchor_local is not None else None),
                     'delta': m, 'delta_big': m_big,
                     'held_out_local': held, 'held_out_pred_err': perr,
                     'pairs': [{'from': a, 'to': b, 'local': l, 'online': t}
                               for a, b, l, t in pairs]})
    rows.sort(key=lambda r: -r['delta']['amp_scaled'])
    print('ledger epoch %s: %d comparable versions; LINEAR=%.0f' % (EPOCH, len(ledger), LINEAR))
    print()
    h = ('%-34s %4s | %7s %7s %8s | %5s %5s %6s %6s %7s %7s %6s | %-7s %s'
         % ('dataset', 'nVer', 'lvCorr', 'lvErr', 'lvAffErr', 'nPair', 'dir', 'ampS',
            'corr', 'rho', 'scale', 'dBig', 'lv435', 'PRED err 441..444'))
    print(h); print('-' * len(h))
    for r in rows:
        dm = r['delta']
        eb = r['held_out_pred_err']
        es = '/'.join(('%d' % eb[c]) if eb.get(c) is not None else '?' for c in ONLINE_441_444)
        print('%-34s %4d | %7.3f %7.0f %8.0f | %5d %5d %6d %6.3f %7.3f %7.1f %6.1f | %-7s %s'
              % (r['dataset'], r['n_versions_matched'], r['level_corr'],
                 r['level_mean_abs_err'], r['level_oracle_affine_resid'],
                 dm['n'], dm['dir_ok'], dm['amp_scaled'], dm['corr'], dm['rho'],
                 dm['scale_ls'], dm['disp_big'],
                 ('%.0f' % r['level_err_at_v435']) if r['level_err_at_v435'] is not None else '-',
                 es))
    print('\nDelta column uses consecutive evaluated versions inside the epoch (%d pairs max).'
          % (max(r['delta']['n'] for r in rows)))
    out = D / 'pub-dataset-ledger-comparison.json'
    out.write_text(json.dumps({
        'ground_truth': str(LEDGER), 'epoch': EPOCH, 'linear_assumed': LINEAR,
        'online_441_444': ONLINE_441_444, 'anchor_version': ANCHOR,
        'ranking': rows,
        'reading': ('lvCorr/lvErr test whether the dataset absolute level tracks online attention '
                    'across versions (strong); ampS/corr/rho/scale test delta geometry. '
                    'lvAffErr is the residual after letting an oracle pick the best affine map, '
                    'so it isolates shape error from scale error.'),
    }, indent=2) + '\n')
    print('saved', out)


if __name__ == '__main__':
    main()
