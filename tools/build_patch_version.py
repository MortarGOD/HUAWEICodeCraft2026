"""Generic text-patch version builder.

The repo's usual pattern appends a `*_core.py` mechanism to a parent. Some changes are single
expressions deep inside large functions instead, so they are expressed as exact-match text
patches. Every patch must match exactly once and the result must parse, so a stale anchor fails
loudly instead of silently building the wrong file.

Usage:
    python tools/build_patch_version.py --spec docs/patches/v503.json
"""
import argparse
import ast
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(spec_path: Path):
    spec = json.loads(spec_path.read_text())
    version = int(spec['version'])
    parent = int(spec['parent'])
    parent_path = ROOT / f'solution/solution_v{parent}.py'
    parent_src = parent_path.read_text()
    actual = hashlib.sha256(parent_src.encode()).hexdigest()
    if spec.get('parent_sha256'):
        assert actual == spec['parent_sha256'], ('parent hash changed', actual)
    source = parent_src
    for i, (old, new) in enumerate(spec['patches']):
        n = source.count(old)
        assert n == 1, f'patch {i} matches {n} times, need exactly 1:\n{old[:120]}'
        source = source.replace(old, new)
    for core in spec.get('append', []):
        core_path = ROOT / core
        assert core_path.exists(), core
        source += '\n\n' + core_path.read_text()
    source += '\n' if not source.endswith('\n') else ''
    ast.parse(source)
    (ROOT / f'solution/solution_v{version}.py').write_text(source)
    staging = ROOT / f'.v{version}_self_check'
    staging.mkdir(exist_ok=True)
    (staging / 'solution.py').write_text(source)
    archive = ROOT / f'solution/submission_v{version}.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        info = zipfile.ZipInfo('solution.py', date_time=(2026, 9, 10, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, source)
    manifest = {'version': version, 'parent': parent, 'parent_sha256': actual,
                'patch_summary': spec.get('summary', ''),
                'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                'archive_members': ['solution.py']}
    (ROOT / f'docs/v{version}-artifacts.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    build(Path(ap.parse_args().spec))
