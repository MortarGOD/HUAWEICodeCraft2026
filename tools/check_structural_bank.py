"""Structural/data-contract tests; do not infer online prediction quality."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rebuild_structural_bank import query_indices, digest
from self_check import _normalize_attention_dataset, _normalize_linear_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', default='local_fitted_models_structural_v3_candidate')
    args = parser.parse_args()
    path = ROOT / args.bank
    torch.set_num_threads(4)
    manifest = json.loads((path / 'manifest.json').read_text())
    for q, k, local in [(16, 2, 8), (16, 2, 16), (64, 1, 8), (8, 8, 8)]:
        for seed in range(20):
            indices = query_indices(q, k, local, torch.Generator().manual_seed(seed))
            assert len(indices.unique()) == local
            assert torch.equal(indices // (q // k), torch.arange(local) // (local // k))
    # Demonstrate the previous first-eight mapping fails this exact invariant.
    assert not torch.equal(torch.arange(8) // 8, torch.arange(8) // 4)
    bank = torch.load(path / 'attn.pt', map_location='cpu', weights_only=False)
    linear = torch.load(path / 'linear.pt', map_location='cpu', weights_only=False)
    _normalize_attention_dataset(bank)
    _normalize_linear_dataset(linear)
    assert len(bank) == 50 and len(linear) == 10
    tensor_count = 0
    mismatched_length_orders = 0
    kv_identity_samples = 0
    for group in bank:
        assert len(group['calib']) == len(group['test']) == 5
        mismatched_length_orders += group['calibration_lengths'] != group['test_lengths']
        if group['source_model'].startswith('Qwen'):
            indices = torch.tensor(group['source_query_indices'])
            assert torch.equal(indices // 8, torch.arange(group['q_num_heads']) // (group['q_num_heads'] // 2))
        for phase in ('calib', 'test'):
            for sample in group[phase]:
                for role in 'qkv':
                    for tensor in sample[role]:
                        assert tensor.device.type == 'cpu' and not tensor.requires_grad
                        assert torch.isfinite(tensor).all()
                        tensor_count += 1
                if group['source_model'].startswith('deepseek'):
                    assert all(torch.equal(a, b) for a, b in zip(sample['k'], sample['v']))
                    kv_identity_samples += 1
    assert kv_identity_samples == 250
    assert mismatched_length_orders >= 45
    assert sum(len(g['test_activation_list']) for g in linear) == 50
    assert digest(path / 'attn.pt') == manifest['attn_sha256']
    assert digest(path / 'linear.pt') == manifest['linear_sha256']
    print(json.dumps(dict(passed=True, attention_cases=250, linear_cases=50,
        checked_tensors=tensor_count, deepseek_identity_samples=kv_identity_samples,
        different_calibration_test_orders=mismatched_length_orders,
        limitation='Structural invariants only; not online score validation.'), indent=2))


if __name__ == '__main__':
    main()
