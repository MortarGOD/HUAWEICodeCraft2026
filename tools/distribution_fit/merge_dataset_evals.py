"""Merge base and extended dataset evaluations, and derive the supervised weight profile offline.

`evaluate_dataset_multi.py` was run twice per dataset (the original 33-version basis and the
64-version ledger change-point extension) to avoid recomputing versions already measured.
This script merges them and, for attention_history_fit_v1, recomputes the weighted mean under
`local-online-weights.json` directly from the per-case qualities, so the second evaluator pass
for that profile is not needed.
"""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'docs/distribution-fit'
BASE = D / 'pub-dataset-eval'
EXT = D / 'pub-dataset-eval-ext'
DSDIR = ROOT / 'docs/papers/hif4-public-references/test/datasets'


def merge(base, ext):
    out = json.loads(base.read_text())
    if ext and ext.exists():
        for v, rec in json.loads(ext.read_text())['results'].items():
            out['results'][v] = rec
    return out


def label_map(dataset):
    """(group_idx, test_idx) -> fit_cases label, mirroring evaluate_dataset_multi's indexing."""
    import torch
    groups = torch.load(dataset, weights_only=True)
    m = {}
    for gi, g in enumerate(groups):
        for i, fc in enumerate(g.get('fit_cases') or []):
            m[(gi, i)] = fc.get('label')
    return m


def reweight(doc, counts, lmap):
    """Recompute mean_quality with an external label->count profile."""
    for v, rec in doc['results'].items():
        if not rec.get('complete'):
            continue
        tw = 0.0
        acc = 0.0
        for c in rec['cases']:
            lbl = lmap.get((c['group'], c['test']))
            w = float(counts.get(lbl, 0.0)) if lbl else 0.0
            acc += c['quality'] * w
            tw += w
        rec['mean_quality_ext'] = (acc / tw) if tw else None
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(D / 'pub-dataset-eval-merged'))
    a = ap.parse_args()
    dest = Path(a.out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    names = sorted({p.stem for p in BASE.glob('*.json')} |
                   {p.stem for p in EXT.glob('*.json')})
    wpath = DSDIR / 'attention_history_fit_v1' / 'local-online-weights.json'
    counts = json.loads(wpath.read_text())
    counts = counts.get('counts', counts)
    for n in names:
        if n.endswith('_localonline'):
            continue  # derived below
        base = BASE / (n + '.json')
        if not base.exists():
            print('skip (no base):', n)
            continue
        doc = merge(base, EXT / (n + '.json'))
        (dest / (n + '.json')).write_text(json.dumps(doc, indent=2) + '\n')
        nv = len(doc['results'])
        nc = sum(1 for r in doc['results'].values() if r.get('complete'))
        print('%-34s versions=%3d complete=%3d' % (n, nv, nc))
        if n == 'attention_history_fit_v1':
            doc2 = json.loads(json.dumps(doc))
            reweight(doc2, counts, label_map(DSDIR / n / 'attn.pt'))
            for rec in doc2['results'].values():
                if rec.get('complete'):
                    rec['mean_quality'] = rec.get('mean_quality_ext')
            (dest / 'attention_history_fit_v1_localonline.json').write_text(
                json.dumps(doc2, indent=2) + '\n')
            print('%-34s derived from reweighted per-case qualities'
                  % 'attention_history_fit_v1_localonline')
    print('wrote', dest)


if __name__ == '__main__':
    main()
