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
    dest = ROOT / 'local_distribution_fit/round033'
    if (dest / 'atoms.json').exists():
        raise SystemExit('round already exists')
    source = next(a for a in json.loads((ROOT / 'local_distribution_fit/round014/atoms.json').read_text())['atoms'] if a['id'] == 'sink-input-seed2026091600-seedplus4-seedplus3')
    path = ROOT / source['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256']
    group = torch.load(path, weights_only=True)
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    for name, gain, phases in (
        ('test-sharp2', 2., ('test',)),
        ('test-sharp4', 4., ('test',)),
        ('both-sharp2', 2., ('calib', 'test')),
        ('both-sharp4', 4., ('calib', 'test')),
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
    prices = json.loads((ROOT / 'docs/distribution-fit/amplitude-separator-prices.json').read_text())
    report = {'round': 33, 'atoms': records, 'required_versions': prices['required_versions'], 'certificate_sha256': prices['certificate_sha256'], 'selection_evidence': 'The verified 555-column v311 direction-only certificate is closest to round014 seedplus3. Test sharper true Q/K with V and decoy seed fixed: historical v311 token polish is sensitive to concentrated attention. Separate test-only drift from matched calibration/test sharpening. This is a mechanism hypothesis, not an inferred online distribution.', 'limits': 'Adaptive support search; screen against current fixed certificate before full 102-version evaluation. Positive screening does not prove joint feasibility.'}
    text = json.dumps(report, indent=2) + '\n'
    (dest / 'atoms.json').write_text(text)
    doc = ROOT / 'docs/distribution-fit/round033'
    doc.mkdir(exist_ok=True)
    (doc / 'atoms.json').write_text(text)
    print('Generated and checked four shared populations')

if __name__ == '__main__':
    main()
