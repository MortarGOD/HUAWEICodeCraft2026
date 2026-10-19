"""Stronger-clean vmean gradient across clean sources.

round049 vmean at 1/2/4 on trace35-group1 gave weak-but-clean differential (+2 amp).
Sweep vmean stronger (8/12/16) on trace35-group1 and trace35-group3 (both clean), and measure
whether a stronger-vmean stays clean AND drives 306->307/283->302 toward +1819/+1830.
No target tuning; predeclared strength gradient; the correlation triangle (283<->303-306 +0.448)
is what the sweep probes.
"""
import hashlib, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
from tools.probe_attention_population import transform_group

COMBO = [
    ("vstr8-g1", "trace35-group1-row_permute", 8),
    ("vstr12-g1", "trace35-group1-row_permute", 12),
    ("vstr16-g1", "trace35-group1-row_permute", 16),
    ("vstr8-g3", "trace35-group3-row_permute", 8),
    ("vstr12-g3", "trace35-group3-row_permute", 12),
    ("vstr16-g3", "trace35-group3-row_permute", 16),
]

@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    src_atoms = {a['id']: a for a in json.loads((ROOT / 'local_distribution_fit/round003/atoms.json').read_text())['atoms']}
    dest = ROOT / 'local_distribution_fit/round051'
    assert not dest.exists()
    dest.mkdir(parents=True)
    doc = ROOT / 'docs/distribution-fit/round051'; doc.mkdir()
    records = []
    for name, src_id, vs in COMBO:
        src = src_atoms[src_id]
        g = torch.load(ROOT / src['path'], weights_only=True)
        seed = 2026092100 + len(records)
        profile = {'vmean': vs, 'qk_encoding_preserved': True}
        grp = transform_group(g, profile, seed)
        out = dest / (name + '.pt'); torch.save(grp, out)
        records.append({'id': name, 'path': str(out.relative_to(ROOT)),
            'sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'seed': seed,
            'mode': 'row_permute', 'fit_eligible': False, 'profile': {'vmean': vs},
            'source_path': src['path'], 'source_sha256': src['sha256']})
    report = {'round': 51, 'atoms': records,
              'mechanism': 'stronger-clean vmean gradient on clean trace sources; measure 306/307,283/302,303/306 first.',
              'limits': 'Controlled mechanism gradient on clean sources; not online distribution evidence; no target tuning.'}
    text = json.dumps(report, indent=2) + '\n'
    (dest / 'atoms.json').write_text(text); (doc / 'atoms.json').write_text(text)
    print('round051 strong-clean vmean atoms:', len(records))

if __name__ == '__main__':
    main()
