"""Fixed shared populations probing logit scale and calibration/test drift."""
import hashlib
import json
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from tools.probe_attention_population import transform_group

def main():
    torch.set_num_threads(2)
    dest = ROOT / 'local_distribution_fit/round025'
    if (dest / 'atoms.json').exists():
        raise SystemExit('round already exists')
    source = next(a for a in json.loads((ROOT / 'local_distribution_fit/round013/atoms.json').read_text())['atoms'] if a['id'] == 'sink-input-seed2026091600-seedplus4')
    path = ROOT / source['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256']
    group = torch.load(path, weights_only=True)
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    for name, gain, phases in (
        ('qk-gain045', .45, ('calib', 'test')),
        ('qk-gain055', .55, ('calib', 'test')),
        ('calib-only-gain05', .5, ('calib',)),
        ('test-only-gain05', .5, ('test',)),
    ):
        transformed = transform_group(group, {'gain': gain}, 2026091800)
        for phase in ('calib', 'test'):
            if phase not in phases:
                transformed[phase] = group[phase]
            else:
                for original, current in zip(group[phase], transformed[phase]):
                    current['v'] = original['v']
        profile = {'gain': gain, 'phases': list(phases), 'v_encoding_preserved': True}
        transformed['population_profile'] = profile
        for phase in ('calib', 'test'):
            for original, current in zip(group[phase], transformed[phase]):
                for a, b in zip(original['v'], current['v']):
                    assert torch.equal(a, b)
                if phase not in phases:
                    assert current is original
        out = dest / (name + '.pt')
        torch.save(transformed, out)
        records.append(dict(source, id=name, path=str(out.relative_to(ROOT)), sha256=hashlib.sha256(out.read_bytes()).hexdigest(), source_path=source['path'], source_sha256=source['sha256'], profile=profile))
    prices = json.loads((ROOT / 'docs/distribution-fit/conflict-prices.json').read_text())
    report = {'round': 25, 'atoms': records, 'required_versions': prices['required_versions'], 'acceptance_relative_tolerance': .15, 'selection_evidence': 'The 500-column certificate is closest to the shared half-QK-scale population. Probe neighboring scales and calibration/test scale drift without changing V, decoy seed, targets, or per-version inputs.', 'limits': 'Adaptive support search; screen against current fixed certificate before full 102-version evaluation. Positive screening does not prove joint feasibility.'}
    text = json.dumps(report, indent=2) + '\n'
    (dest / 'atoms.json').write_text(text)
    doc = ROOT / 'docs/distribution-fit/round025'
    doc.mkdir(exist_ok=True)
    (doc / 'atoms.json').write_text(text)
    print('Generated and checked four shared populations')

if __name__ == '__main__':
    main()
