"""MHA shape pressure by selecting one encoded Q head per original KV group."""
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt', weights_only=False)
    groups = []
    for index in (0, 24, 49):
        group = copy.deepcopy(source[index])
        qh, kh, dim = [group[k] for k in ('q_num_heads', 'kv_num_heads', 'head_dim')]
        assert qh % kh == 0 and dim % 16 == 0
        for sample in group['calib'] + group['test']:
            q, scale = sample['q']
            rows = q.shape[0]
            q = q.reshape(rows, qh, dim)[:, ::qh//kh].contiguous().reshape(rows, kh*dim)
            scale = scale.reshape(rows, qh, dim//16)[:, ::qh//kh].contiguous().reshape(rows, kh*dim//16)
            sample['q'] = (q, scale)
        group['q_num_heads'] = kh
        groups.append(group)
    with tempfile.TemporaryDirectory(prefix='attention-campaign-mha-') as directory:
        torch.save(groups, Path(directory)/'attn.pt')
        for version in (429, 430, 431, 432, 433, 434):
            target = ROOT/f'docs/v{version}-attention-mha.json'
            with target.with_suffix('.log').open('w') as log:
                subprocess.run([sys.executable, str(ROOT/'tools/verify_attention_campaign.py'),
                                '--candidate', str(version), '--dataset', directory,
                                '--output', str(target)], cwd=ROOT, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            record = json.loads(target.read_text())
            record['synthetic_mha_source_groups'] = [0, 24, 49]
            record['construction'] = 'One existing encoded Q head per KV group; unchanged K/V. Reconstructible shape pressure, not sampled online MHA.'
            target.write_text(json.dumps(record, indent=2)+'\n')
            print(version, record['summary'], flush=True)


if __name__ == '__main__':
    main()
