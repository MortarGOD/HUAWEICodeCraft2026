"""Regenerate targets with user-steered 20% amplitude tolerance.
Basis: |delta|>=10 -> 20% ; delta<10 -> 5 points.
Keeps online_total_delta, linear_delta, target, direction rules unchanged.
Writes docs/distribution-fit/targets-20pct.json (strict_tolerance == tolerance)."""
import json, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'docs/distribution-fit'
src = json.loads((D / 'targets-large-first.json').read_text())
es = src['edges']
new = []
for e in es:
    t = abs(e['target'])
    tol = max(t * 0.20, 1.0) if t >= 10 else 5.0
    new.append(dict(e, tolerance=tol, strict_tolerance=tol))
out = dict(src, edges=new, tolerance_basis=(
    'User steering 20%: |delta|>=10 ->20%; delta<10 ->5 points. '
    'All nonzero directions remain strict. Replaces 15%/15/5 basis.'))
tmp = D / 'targets-20pct.json.tmp'
tmp.write_text(json.dumps(out, indent=2) + '\n')
tmp.replace(D / 'targets-20pct.json')
c = collections.Counter(
    'big(>=100)' if abs(e['target']) >= 100
    else 'mid(10-100)' if abs(e['target']) >= 10
    else 'small(<10)' for e in new)
print('targets-20pct.json written; edges:', len(new))
print(dict(c))
old = {(e['from'], e['to']): e['tolerance'] for e in src['edges']}
print('\nold -> new tolerance for 6 largest:')
for e in sorted(new, key=lambda e: -abs(e['target']))[:6]:
    print("  {}->{}: target={:.0f} old_tol={:.1f} -> new_tol={:.1f}".format(
        e['from'], e['to'], e['target'], old[(e['from'], e['to'])], e['tolerance']))
