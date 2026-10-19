"""Mild-strength vmean differential atom on a CLEAN source.

Round032 vmean atoms used the heavy m1-sink source (millions-magnitude, swamping the solver).
Here apply the same vmean mechanism (raw['v'] += vmean * raw['v'][:1]) to the clean round003
trace35-group1 source (max|raw| ~5k), so the differential signal survives without toxic extremes.
The vmean dial (vmean 1,2,4) is predeclared; no target tuning.
"""
import copy, hashlib, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from tools.probe_attention_population import transform_group

SOURCE = 'trace35-group1-row_permute'
GRADIENT = [("vmeanclean-1", 1), ("vmeanclean-2", 2), ("vmeanclean-4", 4)]

@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    src = next(a for a in json.loads((ROOT / 'local_distribution_fit/round003/atoms.json').read_text())['atoms']
               if a['id'] == SOURCE)
    path = ROOT / src['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == src['sha256']
    g = torch.load(path, weights_only=True)
    dest = ROOT / 'local_distribution_fit/round049'
    assert not dest.exists()
    dest.mkdir(parents=True)
    doc = ROOT / 'docs/distribution-fit/round049'
    doc.mkdir()
    records = []
    for name, vmean in GRADIENT:
        seed = 2026091900 + len(records)
        profile = {'vmean': vmean, 'qk_encoding_preserved': True}
        grp = transform_group(g, profile, seed)
        out = dest / (name + '.pt')
        torch.save(grp, out)
        # sanity: non-catastrophic magnitude vs source
        records.append({
            'id': name, 'path': str(out.relative_to(ROOT)),
            'sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'seed': seed,
            'mode': 'row_permute', 'fit_eligible': False,
            'model': src.get('model', 'trace'),
            'profile': {'vmean': vmean, 'qk_encoding_preserved': True},
            'source_path': src['path'], 'source_sha256': src['sha256'],
        })
    report = {'round': 49, 'atoms': records,
              'mechanism': 'mild vmean differential atom on clean trace35-group1 source; measure 306/307 and 303/306 first.',
              'limits': 'Controlled mechanism gradient on clean source; not online distribution evidence; no target tuning.'}
    text = json.dumps(report, indent=2) + '\n'
    (dest / 'atoms.json').write_text(text)
    (doc / 'atoms.json').write_text(text)
    print('round049 vmean-clean atoms:', len(records))

if __name__ == '__main__':
    main()
