"""Exact direction-boundary line searches on fully measured shared columns."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/distribution-fit';sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--paired',action='store_true');parser.add_argument('--transfer',action='store_true');parser.add_argument('--plateau',action='store_true');parser.add_argument('--steps',type=int,default=20);args=parser.parse_args()
    source=D/'direction-transfer-live-checkpoint.json';report=json.loads(source.read_text())
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
    from tools.distribution_fit.fit import simplex
    rows=[];bounds=[]
    for i,d in enumerate(delta):
        if d:
            sign=np.sign(d);rows.append(sign*A[i]);bounds.append(1e-8-sign*linear[i])
        else:
            rows.extend([A[i],-A[i]]);bounds.extend([-1-linear[i],linear[i]-1])
    G=np.asarray(rows);h=np.asarray(bounds);norm=np.linalg.norm(G,axis=1)
    assert not np.any((norm==0)&(h>0))
    active=norm>0;G=G[active]/norm[active,None];h=h[active]/norm[active]
    def check(w):
        v=A@w+linear
        ok=np.where(delta>0,v>0,np.where(delta<0,v<0,np.abs(v)<=1+1e-7))
        return v,ok
    value,ok=check(w);best=w.copy();bestcount=int(ok.sum());history=[]
    x=w.copy();correction=np.zeros(len(h));simplex_correction=np.zeros_like(x)
    for cycle in range(20000):
        for i in range(len(h)):
            y=x+correction[i]*G[i]
            amount=max(h[i]-float(G[i]@y),0.)
            x=y+amount*G[i];correction[i]=-amount
        y=x+simplex_correction;x=simplex(y);simplex_correction=y-x
        if cycle%100==0:
            value,ok=check(x);count=int(ok.sum())
            if count>bestcount:
                bestcount=count;best=x.copy();history.append({'cycle':cycle+1,'passed':count});print(history[-1],flush=True)
            if ok.all():break
    value,ok=check(best)
    out={'method':'Global Dykstra projections onto all direction halfspaces and shared simplex','cycles':cycle+1,'passed':int(ok.sum()),'total':len(edges),'all_directions_passed':bool(ok.all()),'weights':dict(zip(ids,best.tolist())),'history':history,'edges':[dict(e,predicted_total_delta=float(value[i]),direction_passed=bool(ok[i])) for i,e in enumerate(edges)],'parent_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'limits':'No infeasibility claim from nonconvergence. Independent raw-cell verification required for improvements.'}
    (D/'direction-dykstra-fit.json').write_text(json.dumps(out,indent=2)+'\n');print('Final',out['passed'],flush=True)
if __name__=='__main__':main()
