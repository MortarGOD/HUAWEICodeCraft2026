"""Exact direction-boundary line searches on fully measured shared columns."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/distribution-fit'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--paired',action='store_true');parser.add_argument('--transfer',action='store_true');parser.add_argument('--plateau',action='store_true');parser.add_argument('--steps',type=int,default=20);args=parser.parse_args()
    source=D/'direction-first-fit.json';report=json.loads(source.read_text())
    edges=json.loads((D/'targets.json').read_text())['edges']
    assert [(e['from'],e['to']) for e in edges]==[(e['from'],e['to']) for e in report['edges']]
    ids=list(report['weights']);versions={e[k] for e in edges for k in ('from','to')}
    hashes={r['version']:r['source_sha256'] for r in json.loads((D/'targets.json').read_text())['versions']}
    cache={};meta={};columns=[]
    for label in ids:
        rnd,atom,case=label.split('/');test=int(case.split('-')[0][4:]);values={}
        if rnd not in meta:meta[rnd]={a['id']:a for a in json.loads((ROOT/'local_distribution_fit'/rnd/'atoms.json').read_text())['atoms']}
        for v in versions:
            key=(rnd,atom,v)
            if key not in cache:
                c=json.loads((D/rnd/'cells'/f'v{v}--{atom}.json').read_text())
                assert c['complete'] and c['source_sha256']==hashes[v] and c['data_sha256']==meta[rnd][atom]['sha256']
                cache[key]=c
            values[v]=next(c['quality'] for c in cache[key]['cases'] if c['test']==test)*25000
        columns.append([values[e['to']]-values[e['from']] for e in edges])
    A=np.asarray(columns).T;linear=np.array([e['linear_delta'] for e in edges]);delta=np.array([e['online_total_delta'] for e in edges]);w=np.array(list(report['weights'].values()))
    def passing(v):return np.where(delta[:,None]>0,v>1e-8,np.where(delta[:,None]<0,v< -1e-8,np.abs(v)<=1+1e-7))
    def rank(v):
        p=passing(v);return p.sum(0)
    rng=np.random.default_rng(2026092701)
    loss_scale=np.maximum(np.abs(A@w+linear),1e-3)
    def losses(v):
        r=np.where(delta[:,None]>0,np.maximum(1e-6-v,0),np.where(delta[:,None]<0,np.maximum(v+1e-6,0),np.maximum(np.abs(v)-1,0)))
        return ((r/loss_scale[:,None])**2).sum(0)
    history=[]
    for step in range(args.steps):
        base=A@w+linear;current=int(rank(base[:,None])[0]);best=(current,None,None);best_loss=float(losses(base[:,None])[0])
        endpoints=[(j,j,1.) for j in range(A.shape[1])]
        if args.paired:
            endpoints += [(int(j),int(k),float(t)) for j,k,t in zip(rng.integers(A.shape[1],size=6000),rng.integers(A.shape[1],size=6000),rng.choice([.01,.1,.25,.5,.75,.9,.99],size=6000))]
        if args.transfer:
            endpoints=[(j,int(k),-float(w[k])) for k in np.flatnonzero(w>1e-12) for j in range(A.shape[1]) if j!=k]
        protected=passing(base[:,None])[:,0]
        for j,k,mix in endpoints:
            end=base+(-mix)*(A[:,j]-A[:,k]) if mix<0 else A[:,j]*mix+A[:,k]*(1-mix)+linear
            diff=end-base
            # Every pass-count change lies at a zero or tie-band boundary.
            bounds=np.where(delta==0,1.,np.sign(delta)*1e-8)
            valid=np.abs(diff)>1e-30
            cuts=np.concatenate(([0.,1.],((bounds-base)[valid]/diff[valid]),((-bounds-base)[valid]/diff[valid])))
            cuts=np.unique(cuts[(cuts>=0)&(cuts<=1)])
            alphas=np.unique(np.concatenate((cuts,(cuts[1:]+cuts[:-1])/2)))
            pred=base[:,None]+diff[:,None]*alphas
            counts=rank(pred)
            counts=np.where(passing(pred)[protected].all(0),counts,-1)
            error=losses(pred)
            i=int(np.lexsort((error,-counts))[0]) if args.plateau else int(np.argmax(counts))
            if counts[i]>best[0] or (args.plateau and counts[i]==best[0] and error[i]<best_loss-1e-10):
                alpha=float(alphas[i]);trial=w.copy()
                if mix<0:
                    trial[j]+=alpha*(-mix);trial[k]-=alpha*(-mix)
                else:
                    trial*=1-alpha;trial[j]+=alpha*mix;trial[k]+=alpha*(1-mix)
                actual=A@trial+linear
                if int(rank(actual[:,None])[0])!=counts[i] or not passing(actual[:,None])[protected].all():continue
                best=(int(counts[i]),(j,k,mix),alpha);best_loss=float(losses(actual[:,None])[0])
        if best[1] is None:break
        count,endpoint,alpha=best;j,k,mix=endpoint
        if mix<0:
            amount=alpha*(-mix);w[j]+=amount;w[k]-=amount
        else:
            w*=1-alpha;w[j]+=alpha*mix;w[k]+=alpha*(1-mix)
        assert int(rank((A@w+linear)[:,None])[0])==count
        history.append({'step':step+1,'passed':count,'column':ids[j],'second_column':ids[k],'endpoint_mix':mix,'mixture_fraction':alpha,'direction_violation_loss':best_loss});print(history[-1],flush=True)
    total=A@w+linear;passed=passing(total[:,None])[:,0]
    report.update(weights=dict(zip(ids,w.tolist())),passed=int(passed.sum()),all_directions_passed=bool(passed.all()),line_search_history=history,line_search_parent_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    report['edges']=[dict(e,predicted_attention_delta=float((A@w)[i]),predicted_total_delta=float(total[i]),direction_passed=bool(passed[i])) for i,e in enumerate(edges)]
    report['line_search_limits']='All 98 constraints checked on shared weights. Local improvement only, not global optimality or held-out validation.'
    (D/('direction-first-plateau.json' if args.plateau else 'direction-first-transfers.json' if args.transfer else 'direction-first-paired-lines.json' if args.paired else 'direction-first-line-search.json')).write_text(json.dumps(report,indent=2)+'\n')
    print('Final',report['passed'],'/',report['total'],flush=True)
if __name__=='__main__':main()
