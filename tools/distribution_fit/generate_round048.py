"""Fast v307 GPTQ-enable mechanism gradient using the default (non-projected) channel-covariance path.

The checkpoint-projection path (round045) is slow. The default `_attention_sample` path shares a
persistent latent across channels, producing off-diagonal Q/K covariance that GPTQ can fit, so it
should also drive GPTQ gain. This gradient sweeps channel spread / length / context_drift.
No atom is tuned to a target delta; predeclared mechanism gradient. Fast to measure.
"""
import hashlib, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from tools.build_model_proxy_dataset import MODELS, build_attention_group

# (name, source, rows, drift)  -- source='random' uses the fast _attention_sample channel-covariance
GRADIENT = [
    ("fast-ch-l128-d05", 'random', (128,) * 5, 0.05),
    ("fast-ch-l128-d15", 'random', (128,) * 5, 0.15),
    ("fast-ch-l256-d05", 'random', (256,) * 5, 0.05),
    ("fast-ch-l256-d15", 'random', (256,) * 5, 0.15),
    ("fast-ch-l512-d05", 'random', (512,) * 5, 0.05),
]

@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    dest = ROOT / 'local_distribution_fit/round048'
    assert not dest.exists()
    dest.mkdir(parents=True)
    doc = ROOT / 'docs/distribution-fit/round048'
    doc.mkdir()
    records = []
    for name, source, rows, drift in GRADIENT:
        seed = 2026091800 + len(records)
        g = build_attention_group(MODELS[0], seed, 'covariance', rows,
                                  source, ROOT / 'local_model_tensor_cache', drift)
        out = dest / (name + '.pt')
        torch.save(g, out)
        records.append({
            'id': name, 'path': str(out.relative_to(ROOT)),
            'sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'seed': seed,
            'mode': 'correlated_noise', 'noise': .05, 'fit_eligible': False,
            'model': MODELS[0].name,
            'profile': {'source': source, 'rows': rows, 'context_drift': drift}
        })
    report = {'round': 48, 'atoms': records,
              'mechanism': 'v307 GPTQ-enable gain gradient on fast random-source channel covariance; measure 306/307 first.',
              'limits': 'Synthetic fast-channel-covariance gradient; not online distribution evidence.'}
    text = json.dumps(report, indent=2) + '\n'
    (dest / 'atoms.json').write_text(text)
    (doc / 'atoms.json').write_text(text)
    print('round048 fast gradient atoms:', len(records))

if __name__ == '__main__':
    main()
