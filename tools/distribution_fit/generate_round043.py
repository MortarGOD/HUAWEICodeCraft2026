"""Build a controlled v311-mechanism atom family by sweeping the repeated-V tying strength.

Reverse-inference from the v311 code diff (`_polish_value_mean`): it picks, per channel,
the token with the most favorable argmin of (sum e)^2+sum(e^2). When V rows are identical
(tied) the argmin is arbitrary; the polish can pick a position whose shift reduces the
proxy objective but degrades the real attention output. The degradation magnitude scales
with how tied the V rows are (more repeat -> stronger). No atom is tuned to a target delta;
this is a pre-declared mechanism gradient. Each atom is its own .pt data file.
"""
import copy, hashlib, json
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys_path = str(ROOT)
import sys
sys.path.insert(0, sys_path)
from tools.build_fitted_dataset import quantize_nvfp4

# gradient of repeated-V tying: (repeat_fraction, row_noise)
SWEEP = [
    ("rep04-noise06", 0.04, 0.06),
    ("rep06-noise06", 0.06, 0.06),
    ("rep08-noise06", 0.08, 0.06),
    ("rep04-noise10", 0.04, 0.10),
    ("rep06-noise04", 0.06, 0.04),
    ("rep10-noise06", 0.10, 0.06),
    ("rep06-noise08", 0.06, 0.08),
]

def build(rep_frac, row_noise, seed=2026091100):
    g = {'q_num_heads': 1, 'kv_num_heads': 1, 'head_dim': 64, 'calib': [], 'test': []}
    for phase in ['calib', 'test']:
        for i in range(5):
            rng = torch.Generator().manual_seed(seed + i + (100 if phase == 'test' else 0))
            t = 16
            direction = torch.randn(1, 64, generator=rng)
            direction = direction / direction.square().mean().sqrt()
            q = 1.5 * direction.expand(t, -1) + .03 * torch.randn(t, 64, generator=rng)
            k = .1 * torch.randn(t, 64, generator=rng)
            k[0] = 1.5 * direction[0]
            z = torch.randn(t, 64, generator=rng)
            # repeat fraction: first n rows are identical (tied); rest are independent + small noise
            n_rep = max(1, int(round(rep_frac * t)))
            v = torch.empty_like(z)
            base = z[:1]
            for row in range(t):
                if row < n_rep:
                    v[row] = base[0] + row_noise * z[row]
                else:
                    v[row] = z[row]  # independent, helps break ties
            g[phase].append({r: quantize_nvfp4(x) for r, x in zip('qkv', [q, k, v])})
    return g

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    dest = ROOT / 'local_distribution_fit/round043'
    assert not dest.exists()
    dest.mkdir()
    doc = ROOT / 'docs/distribution-fit/round043'
    doc.mkdir()
    atoms = []
    for name, rep_frac, row_noise in SWEEP:
        g = build(rep_frac, row_noise)
        p = dest / (name + '.pt')
        torch.save(g, p)
        for mode in ['paired', 'row_permute']:
            atoms.append({
                'id': name + '-' + mode, 'path': str(p.relative_to(ROOT)),
                'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                'mode': mode, 'fit_eligible': False, 'seed': 2026091100,
                'noise': row_noise,
                'profile': {'tokens': 16, 'repeat_fraction': rep_frac, 'row_noise': row_noise, 'repeated_v': True}
            })
    out = {'round': 43, 'atoms': atoms,
           'mechanism': 'v311 _polish_value_mean argmin-token sensitivity, gradient sweep of repeated-V tying strength',
           'limits': 'Controlled mechanism gradient, not online distribution evidence; measure 310/311 first, then all versions if any strength lands in range.'}
    text = json.dumps(out, indent=2) + '\n'
    (dest / 'atoms.json').write_text(text)
    (doc / 'atoms.json').write_text(text)
    print('round043 swept atoms:', len(atoms))

if __name__ == '__main__':
    main()
