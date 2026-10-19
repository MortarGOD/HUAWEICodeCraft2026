"""Learn a fixed-certificate response surface; propose shared population parameters.

Only measured prices train the model. Predictions never enter the joint fit.
The bounded initial family varies calibration/test QK gain with fixed source,
V encoding, and decoy seed. All versions receive the same proposed population.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'docs/distribution-fit'

def kernel(a, b, length):
    return np.exp(-.5 * ((a[:, None] - b[None, :]) ** 2).sum(-1) / length ** 2)

def main():
    # Recompute from real cells so a changed certificate cannot silently reuse
    # stale labels. price_conflict validates current target/tolerance identity.
    subprocess.run([sys.executable, str(D.parents[1] / 'tools/distribution_fit/price_conflict.py')], check=True, stdout=subprocess.DEVNULL)
    prices = json.loads((D / 'conflict-prices.json').read_text())
    observed = {}
    for item in prices['ranked_cases']:
        observed.setdefault(item['population'], []).append(item)
    records = []
    excluded = []
    for rnd in ('round013', 'round016', 'round020', 'round025', 'round026'):
        manifest = ROOT / 'local_distribution_fit' / rnd / 'atoms.json'
        if not manifest.exists():
            continue
        for atom in json.loads(manifest.read_text())['atoms']:
            if atom['seed'] != 2026090808:
                continue
            if rnd == 'round013':
                gains = [1., 1.]
            elif 'calib_gain' in atom['profile']:
                gains = [atom['profile'][phase + '_gain'] for phase in ('calib', 'test')]
            else:
                p = atom['profile']; phases = p.get('phases', ['calib', 'test'])
                gains = [p['gain'] if phase in phases else 1. for phase in ('calib', 'test')]
            excluded.append(np.log2(gains))
            cases = sorted(observed.get(rnd + '/' + atom['id'], []), key=lambda r:r['test'])
            if len(cases) != 5:
                continue
            records.append({'population':rnd + '/' + atom['id'], 'gains':gains, 'margins':[r['margin'] for r in cases]})
    assert len(records) >= 3, 'Need at least three measured parameter points'
    x = np.log2([r['gains'] for r in records]); y = np.array([r['margins'] for r in records])
    mean = y.mean(0); scale = np.maximum(y.std(0), 1.)
    z = (y - mean) / scale
    best = None
    for length in (.25, .5, 1., 2.):
        for noise in (.001, .01, .1):
            k = kernel(x, x, length) + noise * np.eye(len(x))
            inv = np.linalg.inv(k); alpha = inv @ z
            objective = float((z * alpha).sum() + z.shape[1] * np.linalg.slogdet(k)[1])
            if best is None or objective < best[0]:
                best = (objective, length, noise, inv, alpha)
    _, length, noise, inv, alpha = best
    # Analytic leave-one-out error diagnoses whether this tiny surrogate helps.
    loo = z - alpha / np.diag(inv)[:, None]
    loo_mae = float(np.abs((loo-z)*scale).mean())
    baseline_loo = (y.sum(0)-y)/(len(y)-1)
    baseline_mae = float(np.abs(baseline_loo-y).mean())
    rng = np.random.default_rng(2026092501)
    candidates = rng.uniform(-2, 2, (4096, 2))
    excluded = np.array(excluded)
    candidates = candidates[np.sqrt(((candidates[:,None]-excluded[None])**2).sum(-1)).min(1) > .10]
    kstar = kernel(candidates, x, length)
    pred = (kstar @ alpha)*scale + mean
    sd = np.sqrt(np.maximum(1 - (kstar @ inv*kstar).sum(1), 0))[:,None]*scale
    acquisition = (pred + 2*sd).max(1)
    selected=[]
    for i in np.argsort(-acquisition):
        if any(np.linalg.norm(candidates[i]-candidates[j]) < .35 for j in selected):
            continue
        selected.append(int(i))
        if len(selected)==4: break
    report={'method':'RBF Gaussian process with multi-output measured certificate margins and UCB acquisition', 'certificate_sha256':hashlib.sha256((D/'reduced-conflict-certificate.json').read_bytes()).hexdigest(), 'training':records,'length_scale_log2':length,'noise':noise,'loo_margin_mae':loo_mae,'constant_baseline_loo_mae':baseline_mae,'surrogate_beats_constant_loo':loo_mae<baseline_mae,'proposals':[{'calib_gain':float(2**candidates[i,0]),'test_gain':float(2**candidates[i,1]),'predicted_case_margins':pred[i].tolist(),'case_standard_deviation':sd[i].tolist(),'acquisition':float(acquisition[i])} for i in selected],'limits':'Small adaptive training set; UCB is an exploration heuristic, not calibrated certainty. Predictions do not count as matches. Real measured cells and full joint refit remain mandatory. This first learned generator covers only QK gain and calibration drift, not the whole distribution family.'}
    (D/'gp-scale-proposals.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('loo_margin_mae','constant_baseline_loo_mae','surrogate_beats_constant_loo','proposals')}))

if __name__ == '__main__':
    main()
