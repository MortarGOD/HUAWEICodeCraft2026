"""Read-only identity, small-matrix risk and holdout-coverage audit."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile
import torch

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / 'docs/papers/hif4-public-references/test'


def read_defs(path, names):
    source = path.read_text()
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert set(n.name for n in nodes) == set(names)
    env = {'torch': torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return env


def run():
    torch.set_num_threads(4)
    report = {'commit': '2b8fc7be13c846aeea31ec9fcc197185bf429fd4', 'packages': {}}
    files = {}
    for name in ('attention-v419-v8', 'attention-search-v9'):
        path = REF / 'solution' / name / 'solution.py'
        files[name] = path
        archive = REF / 'solution' / ('submission-' + name + '.zip')
        with zipfile.ZipFile(archive) as z:
            assert z.namelist() == ['solution.py']
            content = z.read('solution.py')
        normalize = lambda b: b.replace(b'\r\n', b'\n')
        assert normalize(content) == normalize(path.read_bytes())
        report['packages'][name] = {
            'git_source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'zip_source_sha256': hashlib.sha256(content).hexdigest(),
            'zip_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
            'normalized_equal': True}
    env = read_defs(files['attention-v419-v8'], ['_attention_informative_holdout', '_value_key_risk_terms'])
    env9 = read_defs(files['attention-search-v9'], ['_value_coordinate_metric'])
    gen = torch.Generator().manual_seed(809)
    residuals = []
    for t in (1, 2, 7, 32):
        for h in (1, 3):
            p = torch.randn(h, 5, t, dtype=torch.float64, generator=gen).softmax(-1)
            variance = torch.rand(h, 5, dtype=torch.float64, generator=gen) * 3
            jacobian = torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)
            expected = p.transpose(-1, -2) @ p / 5
            expected += (variance[:, :, None, None] * (jacobian.transpose(-1, -2) @ jacobian)).mean(1)
            expected += torch.eye(t, dtype=torch.float64) / t ** 2
            actual = env9['_value_coordinate_metric'](p, variance)
            assert torch.allclose(expected, actual, atol=1e-10, rtol=1e-10)
            e = torch.randn(h, t, 9, dtype=torch.float64, generator=gen)
            gradient, curvature = env['_value_key_risk_terms'](e, p, variance)
            assert torch.allclose(gradient, actual @ e, atol=1e-10, rtol=1e-10)
            assert torch.allclose(curvature, actual.diagonal(dim1=-1, dim2=-2), atol=1e-10, rtol=1e-10)
            assert float(torch.linalg.eigvalsh(actual).min()) > 0
            residuals.append(float((actual - expected).abs().max()))
    report['risk_math'] = {'cases': len(residuals), 'max_error': max(residuals),
                           'scope': 'Given nonnegative variance; not proof of fitted uncertainty transfer.'}
    banks = [ROOT / 'mini_sample', ROOT / 'local_fitted_models_attention_feedback_v2',
             ROOT / 'local_fitted_models_covariance_v1',
             REF / 'datasets/qwen3_5_35b_a3b_v1', REF / 'datasets/qwen2_5_14b_online_v1']
    report['holdout_coverage'] = []
    for bank in banks:
        path = bank / 'attn.pt'
        if not path.exists():
            report['holdout_coverage'].append({'bank': str(bank), 'missing': True})
            continue
        groups = torch.load(path, weights_only=True)
        affected = []
        for i, group in enumerate(groups):
            samples = group['calib']
            changed = env['_attention_informative_holdout'](samples, group['q_num_heads'], group['head_dim'])
            if changed is not samples:
                assert all(any(s is old for old in samples) for s in changed)
                assert sorted(map(id, changed)) == sorted(map(id, samples))
                affected.append({'group': i, 'before': [len(s['q'][0]) for s in samples],
                                 'after': [len(s['q'][0]) for s in changed]})
        report['holdout_coverage'].append({'bank': str(bank.relative_to(ROOT)), 'groups': len(groups), 'affected': affected})
    report['complete'] = True
    (ROOT / 'docs/reference-v8-v9-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print({'risk_math': report['risk_math'], 'coverage': [(r['bank'], len(r.get('affected', []))) for r in report['holdout_coverage']]})


if __name__ == '__main__':
    run()
