"""Fine gradient of the round038 all-tied v311 mechanism, dialing row-noise to weaken strength.

Reverse-inference: v311 `_polish_value_mean` degrades real attention when V rows are near-tied
(argmin picks a token whose shift lowers the proxy objective but hurts output). The degradation
strength falls as row-noise rises (rows less tied). round038 noise=0 -> -63344pt, noise=0.03
-> -11620pt. Target 310->311 is -4995. Sweep noise in the weakened band; no atom tuned to a delta.
"""
import copy, hashlib, json
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from tools.build_fitted_dataset import quantize_nvfp4
from hif4_quant import dequantize_nvfp4
from evaluate_local import gqa_attention

SWEEP = [
    ("tie-noise05", 0.05),
    ("tie-noise08", 0.08),
    ("tie-noise12", 0.12),
    ("tie-noise18", 0.18),
    ("tie-noise25", 0.25),
    ("tie-noise35", 0.35),
]

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    dest = ROOT / 'local_distribution_fit/round044'
    assert not dest.exists()
    dest.mkdir()
    doc = ROOT / 'docs/distribution-fit/round044'
    doc.mkdir()
    atoms = []
    for name, noise in SWEEP:
        g = {'q_num_heads': 1, 'kv_num_heads': 1, 'head_dim': 64, 'calib': [], 'test': []}
        for phase in ['calib', 'test']:
            for i in range(5):
                rng = torch.Generator().manual_seed(2026091100 + i + (100 if phase == 'test' else 0))
                t = 16
                direction = torch.randn(1, 64, generator=rng)
                direction = direction / direction.square().mean().sqrt()
                q = 1.5 * direction.expand(t, -1) + .03 * torch.randn(t, 64, generator=rng)
                k = .1 * torch.randn(t, 64, generator=rng)
                k[0] = 1.5 * direction[0]
                z = torch.randn(t, 64, generator=rng)
                v = z[:1].expand(t, -1).clone() + noise * z
                g[phase].append({r: quantize_nvfp4(x) for r, x in zip('qkv', [q, k, v])})
        p = dest / (name + '.pt')
        torch.save(g, p)
        for mode in ['paired']:
            atoms.append({
                'id': name + '-' + mode, 'path': str(p.relative_to(ROOT)),
                'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                'mode': mode, 'fit_eligible': False, 'seed': 2026091100, 'noise': noise,
                'profile': {'tokens': 16, 'all_tied': True, 'row_noise': noise}
            })
    out = {'round': 44, 'atoms': atoms,
           'mechanism': 'v311 all-tied repeated-V strength gradient (row_noise), target band near -4995pt',
           'limits': 'Controlled mechanism gradient, not online distribution evidence; measure 310/311 first.'}
    text = json.dumps(out, indent=2) + '\n'
    (dest / 'atoms.json').write_text(text)
    (doc / 'atoms.json').write_text(text)
    print('round044 sweep atoms:', len(atoms))

if __name__ == '__main__':
    main()
