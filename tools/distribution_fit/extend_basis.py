"""Extend an amplitude basis report with additional round/atom per-test columns (weight 0).

Column label = '<round>/<atom>/test<test>-tokens<tokens>' with test/tokens read from the actual
measured cell signature, so it works for atoms that store rows instead of tokens in their profile.
Then grow_amplitude_constraints.py re-runs feasibility over the enlarged support.
"""
import argparse, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'docs/distribution-fit'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--basis', default='amplitude-20pct-v311-growth.json')
    p.add_argument('--output', default='amplitude-20pct-cov-extended.json')
    p.add_argument('--atoms', default='round045:covs-str8-len128,round045:covs-str16-len512')
    a = p.parse_args()
    base = json.loads((D / a.basis).read_text())
    w = base['weights']
    # must not clash: the grow tool requires columns to be readable from their round's cells
    added = []
    # map version -> source_sha256 from the targets file the basis used
    targets_file = a.basis  # default target is targets-20pct.json; but basis has its own targets_file
    tf = base.get('targets_file', 'targets-20pct.json')
    targets = json.loads((D / tf).read_text())
    hashes = {r['version']: r['source_sha256'] for r in targets['versions']}
    # version set from the basis edges
    versions = {e[k] for e in base['edges'] for k in ('from', 'to')}
    for spec in a.atoms.split(','):
        rnd, atom = spec.split(':')
        # find a measured cell to get the test/token signature
        sample = None
        for v in versions:
            p = D / rnd / 'cells' / ('v%s--%s.json' % (v, atom))
            if p.exists() and json.loads(p.read_text()).get('complete'):
                c = json.loads(p.read_text())
                sample = [(x['test'], x['tokens']) for x in c['cases']]
                break
        if sample is None:
            print('WARNING: no complete cell for', rnd, atom, '-> skipping')
            continue
        for test, tokens in sample:
            label = '{}/{}/test{}-tokens{}'.format(rnd, atom, test, tokens)
            if label not in w:
                w[label] = 0.0
                added.append(label)
    base['weights'] = w
    base['method'] = 'basis extension: added {}; grow over enlarged support'.format(a.atoms)
    base['limits'] = 'Experiment: extend basis with v307/GPTQ covariance-mechanism atoms; not the committed result.'
    out = D / a.output
    (out.with_suffix('.tmp')).write_text(json.dumps(base, indent=2) + '\n')
    (out.with_suffix('.tmp')).replace(out)
    print('added', len(added), 'columns; total', len(w))
    print('sample added:', added[:4])

if __name__ == '__main__':
    main()
