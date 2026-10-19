"""Regression of the diagnostic against an independently executed evaluator."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    diagnostic = json.loads((ROOT/'docs/v324-current-error-split.json').read_text())
    reference = json.loads((ROOT/'docs/v371-screen.json').read_text())
    assert diagnostic['complete'] and reference['complete']
    assert diagnostic['production_unchanged']
    path = str(ROOT/'solution/solution_v324.py')
    assert diagnostic['submission_hashes'][path] == reference['sha256']['324']
    expected = {(r['group'], i): s for r in reference['rows']
                for i, s in enumerate(r['versions']['324']['scores'])}
    assert len(diagnostic['cases']) == len(expected) == 50
    for case in diagnostic['cases']:
        score = 1-case['official_current_mse']/case['baseline_mse']
        assert abs(score-expected[case['group'], case['test']]) < 1e-12
        assert case['evaluation_output_dtype'] == 'torch.bfloat16'
        assert case['normal_equation_relative'] < 1e-9
        assert abs(case['evaluation_identity_error']) < 1e-8*case['total_exact_decoder_mse']
    assert diagnostic['validated_public_returns'] == 60
    assert diagnostic['validated_probe_returns'] == 200
    print('PASS: 50 paired scores, dtype and decomposition identities; 260 format checks recorded')


if __name__ == '__main__':
    main()
