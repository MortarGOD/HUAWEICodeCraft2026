"""Install the audited structural diagnostic bank, preserving the previous bank.

No files are deleted. This is NOT an online-prediction qualification command.
"""
import json
from pathlib import Path

from rebuild_structural_bank import ROOT, digest


def main():
    active = ROOT / 'local_fitted_models_attention_feedback_v2'
    candidate = ROOT / 'local_fitted_models_structural_v3_candidate'
    archive = ROOT / 'local_fitted_archive/attention_feedback_v2_before_structural_20260906'
    report_path = ROOT / 'docs/structural-bank-audit.json'
    report = json.loads(report_path.read_text())
    assert report['complete'] is True
    assert set(report['banks']) == {'old', 'new'}
    for rows in report['banks'].values():
        assert len(rows) == 10
        for row in rows:
            assert len(row['versions']) == 10
            assert all(len(v['scores']) == 5 for v in row['versions'].values())
    assert active.is_dir() and candidate.is_dir() and not archive.exists()
    manifest = json.loads((candidate / 'manifest.json').read_text())
    assert digest(candidate / 'attn.pt') == manifest['attn_sha256']
    assert digest(candidate / 'linear.pt') == digest(active / 'linear.pt') == manifest['linear_sha256']
    assert digest(ROOT / 'tools/rebuild_structural_bank.py') == manifest['generator_sha256']
    report['bank_identity'] = {
        'old': {'archive': str(archive.relative_to(ROOT)), 'attn_sha256': digest(active / 'attn.pt')},
        'new': {'active': str(active.relative_to(ROOT)), 'attn_sha256': manifest['attn_sha256']},
    }
    report['evaluation_count'] = 1000
    report['validation_design'] = 'Retrospective, parameters frozen; not a blind prospective holdout.'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    direction_counts = {}
    for name, edges in report['assessment'].items():
        heldout = [e for e in edges.values() if e['split'] == 'heldout']
        direction_counts[name] = dict(matching=sum(e['direction_match'] for e in heldout), total=len(heldout))
    manifest.update(
        status='active structural diagnostic proxy; online prediction NOT qualified',
        score_prediction_allowed=False,
        behavioral_audit='docs/structural-bank-audit.json',
        behavioral_audit_sha256=digest(report_path),
        retrospective_direction_checks=direction_counts,
        old_bank_archive=str(archive.relative_to(ROOT)),
        evaluated_attention_groups=list(range(10)),
        unevaluated_attention_groups=list(range(10, 50)),
        structural_check=dict(attention_cases=250, linear_cases=50, tensors=3000,
                              deepseek_kv_identity_samples=250, distinct_length_orders=50),
        usage='Mechanism diagnostics and counterexamples only; do not reject online-untested ideas solely by local scores.',
    )
    (candidate / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    archive.parent.mkdir(exist_ok=True)
    active.rename(archive)
    try:
        candidate.rename(active)
    except Exception:
        archive.rename(active)
        raise
    (ROOT / 'docs/active-local-dataset.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(active=str(active), recoverable_archive=str(archive),
                          direction_checks=direction_counts, score_prediction_allowed=False), indent=2))


if __name__ == '__main__':
    main()
