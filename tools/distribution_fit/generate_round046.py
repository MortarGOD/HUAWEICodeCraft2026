"""Finer length/strength gradient of the v307 GPTQ-enable mechanism (round045 follow-up).

round045: len128 -> +57..161pt; len512 -> +4849..5996pt; target 306->307 = +1819pt (tol 364).
Interpolate length/strength in between. No target tuning; predeclared gradient.
"""
import hashlib, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from tools.build_model_proxy_dataset import MODELS, build_attention_group

GRADIENT = [
    ("covs-sr8-l256", 8, 8, (256,) * 5),
    ("covs-sr16-l256", 8, 16, (256,) * 5),
    ("covs-sr32-l256", 8, 32, (256,) * 5),
    ("covs-sr32-l384", 8, 32, (384,) * 5),
    ("covs-sr16-l384", 8, 16, (384,) * 5),
    ("covs-sr64-l128", 8, 64, (128,) * 5),
    ("covs-sr64-l256", 8, 64, (256,) * 5),
]

@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    dest = ROOT / 'local_distribution_fit/round046'
    assert not dest.exists()
    dest.mkdir(parents=True)
    doc = ROOT / 'docs/distribution-fit/round046'
    doc.mkdir()
    records = []
    for name, rank, strength, rows in GRADIENT:
        seed = 2026091600 + len(records)
        g = build_attention_group(MODELS[0], seed, 'covariance', rows,
                                  'checkpoint-projection', ROOT / 'local_model_tensor_cache',
                                  .05, rank, strength)
        out = dest / (name + '.pt')
        torch.save(g, out)
        records.append({
            'id': name, 'path': str(out.relative_to(ROOT)),
            'sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'seed': seed,
            'mode': 'correlated_noise', 'noise': .05, 'fit_eligible': False,
            'model': MODELS[0].name,
            'profile': {'covariance_rank': rank, 'structured_strength': strength, 'rows': rows}
        })
    report = {'round': 46, 'atoms': records,
              'mechanism': 'v307 GPTQ length/strength gradient; measure 306/307 first.',
              'limits': 'Synthetic checkpoint-projected covariance gradient; not online distribution evidence.'}
    text = json.dumps(report, indent=2) + '\n'
    (dest / 'atoms.json').write_text(text)
    (doc / 'atoms.json').write_text(text)
    print('round046 gradient atoms:', len(records))

if __name__ == '__main__':
    main()
