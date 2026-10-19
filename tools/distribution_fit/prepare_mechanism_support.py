"""Extend shared support only after every predeclared mechanism atom is audited."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/distribution-fit'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan_path = DOC / args.round / 'full-matrix-plan.json'
    plan = json.loads(plan_path.read_text())
    atoms = plan['atoms']
    assert atoms and len(atoms) == len(set(atoms))
    source = DOC / args.source
    fit = json.loads(source.read_text())
    audits = {}
    added = []
    # Fail before producing any support if a control or any version is missing.
    for atom in atoms:
        subprocess.run([
            sys.executable, str(ROOT / 'tools/distribution_fit/audit_population.py'),
            '--round', args.round, '--atom', atom, '--integrity-only',
        ], check=True)
        path = DOC / args.round / (atom + '-full-audit.json')
        audit = json.loads(path.read_text())
        assert audit['count'] == plan['versions']
        audits[atom] = hashlib.sha256(path.read_bytes()).hexdigest()
        for test, tokens in audit['case_signature']:
            label = f'{args.round}/{atom}/test{test}-tokens{tokens}'
            if label not in fit['weights']:
                fit['weights'][label] = 0.0
                added.append(label)
    fit['explicit_mechanism_extension'] = {
        'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'audit_sha256': audits,
        'added_columns': added,
        'limits': 'Measured synthetic support, not online prevalence or an accepted fit. Preserve existing seed groups and matched amplitudes.',
    }
    (DOC / args.output).write_text(json.dumps(fit, indent=2) + '\n')
    print(f'Prepared {len(added)} measured zero-weight columns')


if __name__ == '__main__':
    main()
