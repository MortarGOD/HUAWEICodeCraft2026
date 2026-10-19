"""Audit a fully measured shared population against actual source/data files."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/distribution-fit'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', required=True)
    parser.add_argument('--atom', required=True)
    parser.add_argument('--integrity-only', action='store_true')
    args = parser.parse_args()
    manifest = ROOT / 'local_distribution_fit' / args.round / 'atoms.json'
    atoms = json.loads(manifest.read_text())['atoms']
    atom = next(a for a in atoms if a['id'] == args.atom)
    assert atom['fit_eligible'] or args.integrity_only
    assert sha(ROOT / atom['path']) == atom['sha256'], 'Actual data hash changed'
    targets_path = DOC / 'targets.json'
    versions = json.loads(targets_path.read_text())['versions']
    signatures = None
    cell_hashes = {}
    for version in versions:
        number = version['version']
        path = DOC / args.round / 'cells' / f'v{number}--{args.atom}.json'
        cell = json.loads(path.read_text())
        assert cell['complete'], f'Incomplete v{number}'
        assert cell['data_sha256'] == atom['sha256']
        assert sha(ROOT / 'solution' / f'solution_v{number}.py') == version['source_sha256'] == cell['source_sha256']
        signature = [(c['test'], c['tokens']) for c in cell['cases']]
        assert len(signature) == 5 and len(set(signature)) == 5
        assert all(math.isfinite(c['quality']) for c in cell['cases'])
        assert signatures is None or signatures == signature
        signatures = signature
        cell_hashes[str(number)] = sha(path)
    report = {
        'population': args.round + '/' + args.atom,
        'count': len(versions),
        'fit_eligible': atom['fit_eligible'],
        'integrity_only': args.integrity_only,
        'versions': [v['version'] for v in versions],
        'data_sha256': atom['sha256'],
        'manifest_sha256': sha(manifest),
        'targets_sha256': sha(targets_path),
        'case_signature': signatures,
        'cell_sha256': cell_hashes,
        'checks': ['actual source/data hashes', 'complete cells', 'consistent five finite case qualities'],
        'limits': 'Integrity audit of training measurements; not independent distribution validation.',
    }
    output = DOC / args.round / (args.atom + '-full-audit.json')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'{report["population"]}: {len(versions)} complete versions verified')


if __name__ == '__main__':
    main()
