"""Diagnostic only: remove the learned reflectors from the published v422.

This runtime override is not a submission version. It identifies whether the
active reflectors restore the loss from removing the full-width polar map.
"""
import argparse
import torch
from tools import verify_v422_linear as paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='local_fitted_models_attention_feedback_v2')
    parser.add_argument('--groups', default='0')
    parser.add_argument('--output', default='docs/v422-identity-probe.json')
    args = parser.parse_args()
    args.parent, args.candidate, args.threads, args.seed = 421, 422, 4, 20260907422
    original_loader, original_writer = paired.load_solution, paired.write_report

    def load(path):
        module = original_loader(path)
        if path.name == 'solution_v422.py':
            def identity(activation, weight):
                return torch.eye(64, dtype=torch.float32, device=activation.device).expand(
                    int(weight.shape[-1]) // 64, 64, 64
                ).contiguous()
            module._v422_joint_block_transform = identity
        return module

    def write(path, report):
        report['runtime_override'] = {
            'candidate_label': 'v422 with identity block maps; diagnostic only',
            'source_hash_is_base_only': True,
            'operation': '_v422_joint_block_transform returns identity; published files unchanged',
        }
        original_writer(path, report)

    paired.load_solution, paired.write_report = load, write
    try:
        paired.run(args)
    finally:
        paired.load_solution, paired.write_report = original_loader, original_writer


if __name__ == '__main__':
    main()
