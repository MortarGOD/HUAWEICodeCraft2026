"""Compare public reference datasets by how well they reproduce our online feedback.

Two independent questions are scored, because they constrain different things:

  A. LEVEL  - does mean_quality * 25000 land on the online attention level?
              online attention ~= online_total - linear(4450); anchor v435 -> 16854.
              This is set by the dataset's difficulty/sequence-length distribution.
  B. GEOMETRY - do local version-to-version DELTAS track the online deltas?
              Measured on every parent->child edge whose endpoints were evaluated.
              Reported raw and after one global least-squares scale, so a dataset is
              not punished merely for living on a different amplitude scale.

Held-out check: v441..444 are recent versions with known online totals and no parent
edge in the fit manifest, so they are a genuine out-of-sample probe.

Inputs : docs/distribution-fit/pub-dataset-eval/*.json
         docs/distribution-fit/targets-20pct.json
Optional reference row: the in-house fitted distribution, if its artifact is present.
"""
import json
import sys
import numpy as np
from pathlib import Path

D = Path(__file__).resolve().parents[2] / 'docs/distribution-fit'
EV = Path(sys.argv[1]) if len(sys.argv) > 1 else (D / 'pub-dataset-eval')
LINEAR = 4450.0
# online totals from receipts (v435 = v440 = 21304 anchor)
ONLINE_TOTAL = {435: 21304, 440: 21304, 441: 21308, 442: 21305, 443: 21304, 444: 21304}
COLUMNS = (441, 442, 443, 444)


def load_targets():
    t = json.loads((D / 'targets-20pct.json').read_text())
    by_edge = {(e['from'], e['to']): e for e in t['edges']}
    return t['edges'], by_edge


def stats_for(local_by_edge, targets, label):
    """local_by_edge: {(from,to): local_points}. Returns the metrics dict for one basis."""
    recs = []
    for (a, b), e in targets.items():
        if (a, b) in local_by_edge:
            recs.append({'from': a, 'to': b, 'local': float(local_by_edge[(a, b)]),
                         'online': float(e['target']), 'tol': float(e['tolerance'])})
    if not recs:
        return None
    L = np.array([r['local'] for r in recs])
    T = np.array([r['online'] for r in recs])
    tol = np.array([r['tol'] for r in recs])
    thr = np.maximum(tol, np.maximum(np.abs(T) * 0.2, 1.0))
    # one global scale, least squares through the origin
    denom = float((L * L).sum())
    scale = float((L * T).sum() / denom) if denom > 0 else float('nan')
    dir_ok = sum(1 for r in recs if r['online'] == 0
                 or (r['local'] > 0) == (r['online'] > 0))
    amp_raw = int((np.abs(L - T) <= thr).sum())
    amp_scaled = int((np.abs(scale * L - T) <= thr).sum())
    corr = (float(np.corrcoef(L, T)[0, 1])
            if len(recs) > 2 and L.std() > 0 and T.std() > 0 else float('nan'))
    # rank correlation: robust to the few catastrophic outliers in the in-house fit
    spearman = (float(np.corrcoef(np.argsort(np.argsort(L)), np.argsort(np.argsort(T)))[0, 1])
                if len(recs) > 3 and L.std() > 0 and T.std() > 0 else float('nan'))
    # per-edge scale dispersion: how consistently does this dataset's scale transfer?
    nz = [(t, l) for t, l in zip(T, L) if abs(l) > 1.0]
    ratios = np.array([t / l for t, l in nz]) if nz else np.array([])
    disp = (float(np.subtract(*np.percentile(ratios, [75, 25])) /
                  abs(np.median(ratios))) if len(ratios) > 3 and np.median(ratios) != 0
            else float('nan'))
    # same, restricted to the edges that actually carry online score (|target| >= 100)
    big = np.array([t / l for t, l in zip(T, L) if abs(l) > 1.0 and abs(t) >= 100.0])
    disp_big = (float(np.subtract(*np.percentile(big, [75, 25])) / abs(np.median(big)))
                if len(big) > 3 and np.median(big) != 0 else float('nan'))
    big_thr = np.maximum(tol, np.maximum(np.abs(T) * 0.2, 1.0))
    big_mask = np.abs(T) >= 100.0
    amp_big_raw = int((np.abs(L - T)[big_mask] <= big_thr[big_mask]).sum())
    amp_big_scaled = int((np.abs(scale * L - T)[big_mask] <= big_thr[big_mask]).sum())
    return {'dataset': label, 'n_edges': len(recs), 'direction_ok': dir_ok,
            'amp20_raw': amp_raw, 'amp20_scaled': amp_scaled,
            'corr': corr, 'spearman': spearman, 'scale_ls': scale, 'scale_dispersion': disp,
            'scale_dispersion_big': disp_big,
            'amp20_big_raw': amp_big_raw, 'amp20_big_scaled': amp_big_scaled,
            'n_edges_big': int(big_mask.sum()),
            'mean_abs_err': float(np.abs(L - T).mean()),
            'mean_abs_err_scaled': float(np.abs(scale * L - T).mean()),
            'edges': recs}


def evaluate_file(f, targets, by_edge):
    d = json.loads(f.read_text())
    q = {int(k): v['mean_quality'] for k, v in d['results'].items() if v.get('complete')}
    n_eval = len(d['results'])
    # A. level
    levels = {v: q[v] * 25000.0 for v in q}
    anchor = levels.get(435)
    held = {c: (levels[c] - anchor) if (c in levels and anchor is not None) else None
            for c in COLUMNS}
    # B. geometry
    local_by_edge = {}
    for (a, b) in by_edge:
        if a in q and b in q:
            local_by_edge[(a, b)] = (q[b] - q[a]) * 25000.0
    s = stats_for(local_by_edge, targets, f.stem)
    if s is None:
        return None
    # use the anchor v435 plus this dataset's global scale to predict online v441..444
    scale = s['scale_ls']
    pred = {c: (ONLINE_TOTAL[435] + scale * held[c])
            if held.get(c) is not None else None for c in COLUMNS}
    pred_err = {c: (abs(pred[c] - ONLINE_TOTAL[c]) if pred.get(c) is not None else None)
                for c in COLUMNS}
    s.update({'n_versions_evaluated': n_eval, 'n_versions_complete': len(q),
              'attention_level_v435': anchor,
              'online_attention_target': ONLINE_TOTAL[435] - LINEAR,
              'level_abs_err_v435': (abs(anchor - (ONLINE_TOTAL[435] - LINEAR))
                                     if anchor is not None else None),
              'held_out_441_444_local': held,
              'held_out_441_444_pred_total': pred,
              'held_out_441_444_pred_err': pred_err,
              'held_out_441_444_online': {c: ONLINE_TOTAL[c] - ONLINE_TOTAL[435] for c in COLUMNS}})
    return s


def reference_row(targets):
    """The in-house fitted distribution, for context: predicted_total_delta per edge."""
    p = D / 'amplitude-20pct-vmean-growth.json'
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    local_by_edge = {(e['from'], e['to']): e['predicted_total_delta'] for e in d['edges']}
    s = stats_for(local_by_edge, targets, 'OURS(amplitude-20pct-vmean-growth)')
    if s:
        s['n_versions_evaluated'] = None
        s['n_versions_complete'] = None
        s['attention_level_v435'] = None
        s['online_attention_target'] = ONLINE_TOTAL[435] - LINEAR
        s['level_abs_err_v435'] = None
        s['held_out_441_444_local'] = {c: None for c in COLUMNS}
        s['held_out_441_444_online'] = {c: ONLINE_TOTAL[c] - ONLINE_TOTAL[435] for c in COLUMNS}
        s['note'] = 'training-support fit on 675 columns; not an independent dataset'
    return s


def main():
    edges, by_edge = load_targets()
    rows = []
    for f in sorted(EV.glob('*.json')):
        r = evaluate_file(f, by_edge, by_edge)
        if r:
            rows.append(r)
    ref = reference_row(by_edge)
    rows.sort(key=lambda r: (-r['amp20_scaled'], -r['direction_ok'],
                             r['mean_abs_err_scaled']))
    hdr = ('%-34s %4s %4s %5s %5s %6s %6s %7s %7s %7s %9s %9s | %-17s | %s'
           % ('dataset', 'nEd', 'dir', 'ampR', 'ampS', 'corr', 'rho', 'scaleLS', 'disp',
              'dispBig', 'lv435', '|lv-tgt|', 'LOCAL d 441..444', 'PRED err 441..444'))
    print('   online: level target %d | held-out local +4/+1/0/0 | actual totals %s'
          % (ONLINE_TOTAL[435] - LINEAR, [ONLINE_TOTAL[c] for c in COLUMNS]))
    print(hdr)
    print('-' * len(hdr))
    order = rows + ([ref] if ref else [])
    for r in order:
        h = r['held_out_441_444_local']
        hs = '/'.join(('%+.0f' % h[c]) if h.get(c) is not None else '  ?' for c in COLUMNS)
        e = r.get('held_out_441_444_pred_err', {})
        es = '/'.join(('%d' % e[c]) if e.get(c) is not None else '?' for c in COLUMNS)
        lv = r['attention_level_v435']
        print('%-34s %4d %4d %5d %5d %6.3f %6.3f %7.3f %7.3f %7.3f %9s %9s | %-17s | %s'
              % (r['dataset'], r['n_edges'], r['direction_ok'], r['amp20_raw'],
                 r['amp20_scaled'], r['corr'], r['spearman'], r['scale_ls'],
                 r['scale_dispersion'], r.get('scale_dispersion_big', float('nan')),
                 ('%.0f' % lv) if lv else '-',
                 ('%.0f' % r['level_abs_err_v435']) if r['level_abs_err_v435'] is not None else '-',
                 hs, es))
    out = D / 'pub-dataset-comparison.json'
    out.write_text(json.dumps({
        'ranking': rows,
        'reference_row': ref,
        'online_attention_target': ONLINE_TOTAL[435] - LINEAR,
        'linear_assumed': LINEAR,
        'online_totals': ONLINE_TOTAL,
        'metric': ('local delta = (mean_quality[to]-mean_quality[from])*25000 on the evaluated '
                   'version set; amp20_scaled applies one global least-squares scale first; '
                   'level = mean_quality[v435]*25000 vs online attention target; '
                   'held-out = v441..444 vs v435'),
        'limits': ('Only edges whose both endpoints are inside the 33-version evaluation set are '
                   'scored; datasets with few test cases give a coarse mean_quality. Level match '
                   'and delta geometry are separate claims and neither alone proves online transfer.'),
    }, indent=2) + '\n')
    print('\nsaved', out)


if __name__ == '__main__':
    main()
