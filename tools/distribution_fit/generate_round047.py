"""Length sweep between 128 and 256 for the v307 GPTQ-enable mechanism.

round046: len128 -> +57..161pt, len256 -> +3326..4232pt; target 306->307 = +1819pt.
The task is finding the length where GPTQ gain lands near target. Sweep mid lengths + drop
structured_strength to flatten the quantizer jump. No target tuning.
"""
import hashlib, json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from tools.build_model_proxy_dataset import MODELS, build_attention_group

GRADIENT = [
    ("covs-s1-l192", 8, 4, (192,) * 5),
    ("covs-s1-l224", 8, 4, (224,) * 5),
    ("covs-s1-l256", 8, 4, (256,) * 5),
    ("covs-s2-l192", 8, 8, (192,) * 5),
    ("covs-s4-l160", 8, 16, (160,) * 5),
    ("covs-s4-l192", 8, 16, (192,) * 5),
    ("covs-s4-l224", 8, 16, (224,) * 5),
]

@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    dest = ROOT / 'local_distribution_fit/round047'
    assert not dest.exists()
    dest.mkdir(parents=True)
    doc = ROOT / 'docs/distribution-fit/round047'
    doc.mkdir()
    records = []
    for name, rank, strength, rows in GRADIENT:
        seed = 2026091700 + len(records)
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
    report = {'round': 47, 'atoms': records,
              'mechanism': 'v307 GPTQ length sweep 128..256; measure 306/307 first.',
              'limits': 'Synthetic checkpoint-projected covariance gradient; not online distribution evidence.'}
    text = json.dumps(report, indent=2) + '\n'
    (dest / 'atoms.json').write_text(text)
    (doc / 'atoms.json').write_text(text)
    print('round047 sweep atoms:', len(records))

if __name__ == '__main__':
    main()
