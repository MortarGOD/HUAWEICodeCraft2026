"""Per-case signal analysis: how much does ONE test case of a dataset tell us about online deltas?

Pooling a dataset into mean_quality hides its internal disagreement. Two reference datasets
(public_quantizer and v419_mini_sample) share a shape profile but are independent draws; their
pooled correlations differ wildly, which can only be sampling noise. This script measures that
noise directly:

  * for every individual test case, correlate its own version-delta series with the online
    targets over the covered edges;
  * report the distribution of those per-case correlations;
  * report the split-half reliability: correlate the pooled series built from a random half of
    the cases with the pooled series from the other half (how reproducible the dataset's own
    signal is), and with the online targets.

A dataset is only a trustworthy online proxy if its per-case correlations agree in sign and if
its split-half reliability is high.
"""
import json
import random
import sys
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parents[2] / 'docs/distribution-fit'
EV = Path(sys.argv[1]) if len(sys.argv) > 1 else (D / 'pub-dataset-eval')
RNG = random.Random(20260910)


def build_case_series(path):
    """case_key -> {version: quality}, plus case weights."""
    d = json.loads(path.read_text())
    series = {}
    for v, rec in d['results'].items():
        if not rec.get('complete'):
            continue
        for c in rec['cases']:
            k = (c['group'], c['test'])
            series.setdefault(k, {})[int(v)] = c['quality']
    weights = {}
    for v, rec in d['results'].items():
        if not rec.get('complete'):
            continue
        for c in rec['cases']:
            weights[(c['group'], c['test'])] = c['weight']
        break
    return series, weights


def main():
    targets = json.loads((D / 'targets-20pct.json').read_text())['edges']
    by_edge = {(e['from'], e['to']): float(e['target']) for e in targets}
    out = []
    for f in sorted(EV.glob('*.json')):
        series, weights = build_case_series(f)
        if not series:
            continue
        vers = sorted(set.intersection(*[set(s) for s in series.values()]))
        edges = [(a, b) for (a, b) in by_edge if a in vers and b in vers]
        if len(edges) < 5 or not series:
            continue
        T = np.array([by_edge[e] for e in edges])
        # per-case correlation
        corrs = []
        for k, s in series.items():
            L = np.array([(s[b] - s[a]) * 25000.0 for a, b in edges])
            if L.std() > 0:
                corrs.append(float(np.corrcoef(L, T)[0, 1]))
        corrs = np.array(corrs)
        # pooled series with the dataset's own weights
        pooled = {}
        for k, s in series.items():
            w = float(weights.get(k, 1.0))
            for v, q in s.items():
                pooled[v] = pooled.get(v, 0.0) + w * q
        tw = sum(float(weights.get(k, 1.0)) for k in series)
        Lp = np.array([(pooled[b] - pooled[a]) / tw * 25000.0 for a, b in edges])
        pooled_corr = float(np.corrcoef(Lp, T)[0, 1]) if Lp.std() > 0 else float('nan')
        # split-half reliability of the dataset's own signal
        keys = list(series)
        rel = []
        for _ in range(200):
            RNG.shuffle(keys)
            h1, h2 = keys[:len(keys) // 2], keys[len(keys) // 2:]
            if not h1 or not h2:
                continue
            def pool(ks):
                p = {}
                for k in ks:
                    for v, q in series[k].items():
                        p[v] = p.get(v, 0.0) + q
                return p
            p1, p2 = pool(h1), pool(h2)
            L1 = np.array([(p1[b] - p1[a]) / len(h1) * 25000.0 for a, b in edges])
            L2 = np.array([(p2[b] - p2[a]) / len(h2) * 25000.0 for a, b in edges])
            if L1.std() > 0 and L2.std() > 0:
                rel.append(float(np.corrcoef(L1, L2)[0, 1]))
        out.append({
            'dataset': f.stem, 'n_cases': len(series), 'n_edges': len(edges),
            'pooled_corr': pooled_corr,
            'percase_corr_median': float(np.median(corrs)),
            'percase_corr_p25': float(np.percentile(corrs, 25)),
            'percase_corr_p75': float(np.percentile(corrs, 75)),
            'percase_frac_positive': float((corrs > 0).mean()),
            'percase_frac_gt_0.5': float((corrs > 0.5).mean()),
            'split_half_reliability': float(np.median(rel)) if rel else float('nan'),
        })
    out.sort(key=lambda r: -r['split_half_reliability'])
    hdr = ('%-34s %6s %5s %8s | %8s %8s %8s | %7s %7s | %8s'
           % ('dataset', 'cases', 'edges', 'pooled', 'case_p25', 'case_med', 'case_p75',
              'frac>0', 'frac>.5', 'splithalf'))
    print(hdr)
    print('-' * len(hdr))
    for r in out:
        print('%-34s %6d %5d %8.3f | %8.3f %8.3f %8.3f | %7.2f %7.2f | %8.3f'
              % (r['dataset'], r['n_cases'], r['n_edges'], r['pooled_corr'],
                 r['percase_corr_p25'], r['percase_corr_median'], r['percase_corr_p75'],
                 r['percase_frac_positive'], r['percase_frac_gt_0.5'],
                 r['split_half_reliability']))
    dest = D / 'pub-dataset-per-case-signal.json'
    dest.write_text(json.dumps({
        'rows': out,
        'reading': ('split_half_reliability is the median correlation between two disjoint halves of '
                    'the dataset\'s own cases; it is an upper bound on how well that dataset can '
                    'predict anything, because it measures self-consistency, not online truth.'),
    }, indent=2) + '\n')
    print('\nsaved', dest)


if __name__ == '__main__':
    main()
