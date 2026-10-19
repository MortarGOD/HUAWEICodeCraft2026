"""Measure v441-v444 (and v435 as anchor) on the 40 atoms of the fitted scoring keys."""
import json, subprocess, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
ROOT = Path('.').resolve(); D = ROOT / 'docs/distribution-fit'
keys = json.loads((D / 'score-keys.json').read_text())
atoms = keys['atoms']
VERS = [435, 441, 442, 443, 444]
def complete(v, rnd, atom):
    p = D / rnd / 'cells' / ('v%s--%s.json' % (v, atom))
    return p.exists() and json.loads(p.read_text()).get('complete')
tasks = []
for v in VERS:
    for spec in atoms:
        rnd, atom = spec.split('/')
        if not complete(v, rnd, atom):
            tasks.append((v, rnd, atom))
print('pending cells:', len(tasks), flush=True)
def run(t):
    v, rnd, atom = t
    try:
        p = subprocess.run([sys.executable, 'tools/distribution_fit/evaluate_cell.py',
                            '--version', str(v), '--atom', atom, '--round', rnd],
                           cwd=str(ROOT), capture_output=True, timeout=120)
        return 'ok' if p.returncode == 0 else 'err'
    except subprocess.TimeoutExpired:
        return 'timeout'
with ThreadPoolExecutor(max_workers=5) as ex:
    futs = {ex.submit(run, t): t for t in tasks}
    done = 0
    for f in as_completed(futs):
        done += 1
        if done % 20 == 0 or f.result() != 'ok':
            print(f.result(), futs[f], f'({done}/{len(tasks)})', flush=True)
print('SCORE_KEYS_DONE', flush=True)
