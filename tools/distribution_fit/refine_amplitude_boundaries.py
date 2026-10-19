"""Exact direction-boundary line searches on fully measured shared columns."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/distribution-fit';sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--paired',action='store_true');parser.add_argument('--transfer',action='store_true');parser.add_argument('--plateau',action='store_true');parser.add_argument('--steps',type=int,default=20);args=parser.parse_args()
    source=D/'amplitude-hard-directions-fit.json';report=json.loads(source.read_text())
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
    tol=np.array([e['tolerance'] for e in edges]);lo=delta-tol;hi=delta+tol
    def directions(v):return np.where(delta[:,None]>0,v>1e-8,np.where(delta[:,None]<0,v< -1e-8,np.abs(v)<=1+1e-7)).all(0)
    def counts(v):return (np.abs(v-delta[:,None])<=tol[:,None]+1e-7).sum(0)
    history=[]
    for step in range(30):
        base=A@w+linear;current=int(counts(base[:,None])[0]);best=(current,None,None)
        for j in range(len(w)):
            diff=A[:,j]+linear-base;valid=np.abs(diff)>1e-30
            bounds=[lo,hi,np.where(delta>0,1e-8,np.where(delta<0,-1e-8,1.)),np.where(delta==0,-1.,0.)]
            cuts=np.unique(np.concatenate(([0.,1.],*[(b-base)[valid]/diff[valid] for b in bounds])))
            cuts=cuts[(cuts>=0)&(cuts<=1)]
            alpha=np.unique(np.r_[cuts,(cuts[1:]+cuts[:-1])/2]);value=base[:,None]+diff[:,None]*alpha
            score=np.where(directions(value),counts(value),-1)
            for i in np.argsort(-score):
                if score[i]<=best[0]:break
                trial=w*(1-alpha[i]);trial[j]+=alpha[i];actual=A@trial+linear
                n=int(counts(actual[:,None])[0])
                if directions(actual[:,None])[0] and n>best[0]:best=(n,j,float(alpha[i]));break
        if best[1] is None:break
        n,j,a=best;w*=1-a;w[j]+=a;history.append({'step':step+1,'amplitude_matched':n,'column':ids[j],'mixture_fraction':a});print(history[-1],flush=True)
    value=A@w+linear;assert directions(value[:,None])[0]
    out={'stage':'amplitude_with_hard_directions','method':'Amplitude interval boundary line search with all directions protected','passed':98,'total':98,'amplitude_matched':int(counts(value[:,None])[0]),'weights':dict(zip(ids,w.tolist())),'history':history,'edges':[dict(e,predicted_total_delta=float(value[i]),direction_passed=True,amplitude_passed=bool(abs(value[i]-delta[i])<=tol[i]+1e-7)) for i,e in enumerate(edges)],'parent_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'limits':'Local shared-mixture improvement only; raw independent verification required.'}
    (D/'amplitude-boundary-fit.json').write_text(json.dumps(out,indent=2)+'\n');print('Final',out['amplitude_matched'],flush=True)
if __name__=='__main__':main()
