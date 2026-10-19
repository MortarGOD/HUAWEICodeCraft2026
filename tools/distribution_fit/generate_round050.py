"""Broader clean differential atom family: (vmean or qk-gain) x (clean trace sources).

Round-7 showed vmean on trace35-group1 (clean) gives +2 amplitude while avoiding toxic magnitudes.
Extend: apply vmean and qk-gain dials to the other clean trace35/trace25 row_permute sources. All these
sources are well-behaved (clean magnitude). No target tuning; predeclared mechanism gradient.
"""
import copy, hashlib, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from tools.probe_attention_population import transform_group

# (name, source_atom, profile)
COMBOS = [
    ("qkclean-g125", "trace35-group3-row_permute", {"gain": 1.25, "qk_encoding_preserved": True}),
    ("qkclean-g050", "trace35-group3-row_permute", {"gain": 0.5, "qk_encoding_preserved": True}),
    ("vmeancl3-4", "trace35-group3-row_permute", {"vmean": 4, "qk_encoding_preserved": True}),
    ("vmeancl7-4", "trace35-group7-row_permute", {"vmean": 4, "qk_encoding_preserved": True}),
    ("vmeancl9-4", "trace35-group9-row_permute", {"vmean": 4, "qk_encoding_preserved": True}),
    ("vmeancl25-4", "trace25-group1-row_permute", {"vmean": 4, "qk_encoding_preserved": True}),
    ("qkclean-g200", "trace25-group1-row_permute", {"gain": 2.0, "qk_encoding_preserved": True}),
]

@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    src_atoms = {a['id']: a for a in json.loads((ROOT / 'local_distribution_fit/round003/atoms.json').read_text())['atoms']}
    dest = ROOT / 'local_distribution_fit/round050'
    assert not dest.exists()
    dest.mkdir(parents=True)
    doc = ROOT / 'docs/distribution-fit/round050'
    doc.mkdir()
    records = []
    for name, source_atom, profile in COMBOS:
        src = src_atoms[source_atom]
        path = ROOT / src['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == src['sha256']
        g = torch.load(path, weights_only=True)
        seed = 2026092000 + len(records)
        grp = transform_group(g, profile, seed)
        out = dest / (name + '.pt')
        torch.save(grp, out)
        records.append({
            'id': name, 'path': str(out.relative_to(ROOT)),
            'sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'seed': seed,
            'mode': 'row_permute', 'fit_eligible': False,
            'profile': profile, 'source_path': src['path'], 'source_sha256': src['sha256'],
        })
    report = {'round': 50, 'atoms': records,
              'mechanism': 'clean differential atom family: (vmean|qk-gain) x clean trace sources; measure 306/307,303/305,303/306,283/302 first.',
              'limits': 'Controlled mechanism gradient on clean sources; not online distribution evidence; no target tuning.'}
    text = json.dumps(report, indent=2) + '\n'
    (dest / 'atoms.json').write_text(text)
    (doc / 'atoms.json').write_text(text)
    print('round050 clean differential atoms:', len(records))

if __name__ == '__main__':
    main()
