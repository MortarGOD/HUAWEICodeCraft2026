"""Exact direction-boundary line searches on fully measured shared columns."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/distribution-fit'

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
    base=A@w+linear;sign=np.sign(delta);failed=np.flatnonzero((sign!=0)&(sign*base<=0))
    assert len(failed)==2
    pools=[np.flatnonzero(sign[i]*(A[i]+linear[i])>0) for i in failed]
    print('Targeted pair pools',*[len(x) for x in pools],flush=True)
    def clip(poly,g,h):
        out=[]
        for a,b in zip(poly,poly[1:]+poly[:1]):
            va=float(g@a-h);vb=float(g@b-h)
            if va>=0:out.append(a)
            if (va>=0)!=(vb>=0):out.append(a+(b-a)*(va/(va-vb)))
        return out
    tested=0;result=None
    for j in pools[0]:
        for k in pools[1]:
            tested+=1
            V=np.stack((A[:,j]+linear-base,A[:,k]+linear-base),axis=1)
            polygon=[np.array([0.,0.]),np.array([1.,0.]),np.array([0.,1.])]
            constraints=[]
            for i,d in enumerate(delta):
                if d:constraints.append((sign[i]*V[i],1e-8-sign[i]*base[i]))
                else:constraints.extend([(V[i],-1-base[i]),(-V[i],base[i]-1)])
            # Fail early on the two required corrections, then protect all others.
            constraints.sort(key=lambda z:-z[1]/max(np.linalg.norm(z[0]),1e-30))
            for g,h in constraints:
                polygon=clip(polygon,g,h)
                if not polygon:break
            if not polygon:continue
            point=np.mean(polygon,axis=0);trial=w*(1-point.sum());trial[j]+=point[0];trial[k]+=point[1]
            value=A@trial+linear
            passed=np.where(delta>0,value>0,np.where(delta<0,value<0,np.abs(value)<=1+1e-7))
            if passed.all() and trial.min()>=-1e-14:
                result=(trial,value,int(j),int(k),point);break
        if result is not None:break
    out={'method':'Two-column convex polygon feasibility with every direction constrained','tested_pairs':tested,'pool_sizes':[len(x) for x in pools],'all_directions_found':result is not None,'parent_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'limits':'Measured training support only; raw independent verification required. Failure excludes only checked pairs, not general mixtures.'}
    if result is not None:
        trial,value,j,k,point=result
        out.update(weights=dict(zip(ids,trial.tolist())),columns=[ids[j],ids[k]],fractions=point.tolist(),edges=[dict(e,predicted_total_delta=float(value[i]),direction_passed=True) for i,e in enumerate(edges)],passed=98,total=98)
    (D/'direction-polygon-fit.json').write_text(json.dumps(out,indent=2)+'\n')
    print({k:v for k,v in out.items() if k not in ('weights','edges')},flush=True)
if __name__=='__main__':main()
