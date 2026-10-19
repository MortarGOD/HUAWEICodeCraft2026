"""Learn a fixed-certificate response surface; propose shared population parameters.

Only measured prices train the model. Predictions never enter the joint fit.
The bounded initial family varies calibration/test QK gain with fixed source,
V encoding, and decoy seed. All versions receive the same proposed population.
"""
import argparse
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
    parser=argparse.ArgumentParser();parser.add_argument('--certificate',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    # Recompute from real cells so a changed certificate cannot silently reuse
    # stale labels. price_conflict validates current target/tolerance identity.
    subprocess.run([sys.executable, str(ROOT / 'tools/distribution_fit/price_amplitude_separator.py'), '--certificate', args.certificate], check=True, stdout=subprocess.DEVNULL)
    prices = json.loads((D / 'amplitude-separator-prices.json').read_text())
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
    assert len(records)>=5, 'Need five measured parameter points for nested validation'
    x=np.log2([r['gains'] for r in records]);y=np.array([r['margins'] for r in records])
    assert len(np.unique(x,axis=0))==len(x), 'Do not leak duplicate parameter points across validation'
    grid=[(l,n) for l in (.25,.5,1.,2.) for n in (.01,.1,1.)]
    def predict(train_x,train_y,test_x,hp,with_sd=False):
        mean=train_y.mean(0);scale=np.maximum(train_y.std(0),1.)
        k=kernel(train_x,train_x,hp[0])+hp[1]*np.eye(len(train_x))
        cross=kernel(test_x,train_x,hp[0]);alpha=np.linalg.solve(k,(train_y-mean)/scale)
        pred=cross@alpha*scale+mean
        if not with_sd:return pred
        variance=np.maximum(1-np.sum(cross*np.linalg.solve(k,cross.T).T,axis=1),0)
        return pred,np.sqrt(variance)[:,None]*scale
    def select(tx,ty):
        errors=[]
        for hp in grid:
            folds=[]
            for j in range(len(tx)):
                keep=np.arange(len(tx))!=j
                folds.append(np.abs(predict(tx[keep],ty[keep],tx[j:j+1],hp)[0]-ty[j]).mean())
            errors.append(float(np.mean(folds)))
        return grid[int(np.argmin(errors))]
    held=[];baseline=[];folds=[]
    for j in range(len(x)):
        keep=np.arange(len(x))!=j;hp=select(x[keep],y[keep])
        pred=predict(x[keep],y[keep],x[j:j+1],hp)[0];base=y[keep].mean(0)
        held.append(np.abs(pred-y[j]).mean());baseline.append(np.abs(base-y[j]).mean())
        folds.append({'population':records[j]['population'],'hyperparameters':hp,'actual':y[j].tolist(),'predicted':pred.tolist(),'baseline':base.tolist()})
    mae=float(np.mean(held));base_mae=float(np.mean(baseline));eligible=mae<.9*base_mae
    hp=select(x,y);proposals=[]
    if eligible:
        rng=np.random.default_rng(2026092502);pool=rng.uniform(-2,2,(4096,2));ex=np.asarray(excluded)
        pool=pool[np.sqrt(((pool[:,None]-ex[None])**2).sum(-1)).min(1)>.10]
        pred,sd=predict(x,y,pool,hp,True);acq=(pred+2*sd).max(1);chosen=[]
        for i in np.argsort(-acq):
            if any(np.linalg.norm(pool[i]-pool[j])<.35 for j in chosen):continue
            chosen.append(int(i))
            if len(chosen)==2:break
        proposals=[{'calib_gain':float(2**pool[i,0]),'test_gain':float(2**pool[i,1]),'predicted_case_margins':pred[i].tolist(),'uncalibrated_sd':sd[i].tolist()} for i in chosen]
    report={'method':'RBF GP with nested parameter-point leave-one-out validation; training-only standardization','certificate_sha256':hashlib.sha256((D/args.certificate).read_bytes()).hexdigest(),'training':records,'nested_folds':folds,'heldout_mae':mae,'constant_baseline_mae':base_mae,'eligible_for_guidance':eligible,'hyperparameters':hp,'proposals':proposals,'limits':'Small fixed-source QK scale family only. Validation tests surrogate parameter interpolation, not online generalization. Predictions never enter fitted cells; UCB uncertainty is not calibrated.'}
    (D/args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('heldout_mae','constant_baseline_mae','eligible_for_guidance','proposals')}))

if __name__=='__main__':main()
