"""Stage one: shared-mixture online direction feasibility, including small moves.

Amplitude fitting is deferred until every direction passes. Exact online ties
retain the existing one-point tolerance. Linear deltas are measured corrections.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.distribution_fit.fit import solve
D=ROOT/'docs/distribution-fit'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scale',choices=['online','uniform','row'],default='online');args=parser.parse_args()
    old=json.loads((D/'case-fit.json').read_text())
    target=json.loads((D/'targets.json').read_text());edges=target['edges'];ids=old['population_ids']
    sources={v['version']:v['source_sha256'] for v in target['versions']}
    cache={};columns=[]
    for label in ids:
        rnd,atom,test=label.split('/');j=int(test.split('-')[0][4:])
        vals={}
        for v in sources:
            key=(rnd,atom,v)
            if key not in cache:
                c=json.loads((D/rnd/'cells'/f'v{v}--{atom}.json').read_text())
                assert c['complete'] and c['source_sha256']==sources[v]
                meta=next(a for a in json.loads((ROOT/'local_distribution_fit'/rnd/'atoms.json').read_text())['atoms'] if a['id']==atom)
                assert c['data_sha256']==meta['sha256']
                cache[key]=c
            vals[v]=next(c['quality'] for c in cache[key]['cases'] if c['test']==j)*25000
        columns.append([vals[e['to']]-vals[e['from']] for e in edges])
    A=np.array(columns).T
    delta=np.array([e['online_total_delta'] for e in edges]);linear=np.array([e['linear_delta'] for e in edges])
    # Finite opposite bounds are exact extrema of the current simplex support.
    low=A.min(1)-1;high=A.max(1)+1
    low=np.where(delta>0,1e-4-linear,low);high=np.where(delta<0,-1e-4-linear,high)
    low=np.where(delta==0,-1-linear,low);high=np.where(delta==0,1-linear,high)
    scale=np.maximum(np.abs(delta),1.) if args.scale=='online' else np.ones(len(edges)) if args.scale=='uniform' else np.maximum(np.linalg.norm(A,axis=1),1.)
    prior=D/'direction-first-fit.json'
    seed=json.loads(prior.read_text()) if prior.exists() else old
    w=np.array([seed['weights'].get(k,0) for k in ids]);history=[];best=None
    for iteration in range(10):
        w,n=solve(A,np.zeros(len(edges)),scale,low,high,w)
        total=A@w+linear
        passed=np.where(delta>0,total>0,np.where(delta<0,total<0,np.abs(total)<=1+1e-7))
        loss=float(np.sum((np.maximum(low-A@w,0)+np.maximum(A@w-high,0))**2/scale**2))
        rank=(int(passed.sum()),-loss)
        if best is None or rank>best[0]:best=(rank,w.copy())
        history.append({'round':iteration+1,'iterations':n,'passed':int(passed.sum())})
        print(history[-1],flush=True)
        if passed.all():break
    w=best[1];total=A@w+linear
    passed=np.where(delta>0,total>0,np.where(delta<0,total<0,np.abs(total)<=1+1e-7))
    detail=[dict(e,predicted_attention_delta=float((A@w)[i]),predicted_total_delta=float(total[i]),direction_passed=bool(passed[i])) for i,e in enumerate(edges)]
    report={'stage':'direction_first','row_scaling':args.scale,'all_directions_passed':bool(passed.all()),'passed':int(passed.sum()),'total':len(edges),'nonzero_directions':int((delta!=0).sum()),'exact_ties':int((delta==0).sum()),'complete_population_columns':len(ids),'history':history,'weights':dict(zip(ids,w.tolist())),'edges':detail,'source_fit_sha256':hashlib.sha256((D/'case-fit.json').read_bytes()).hexdigest(),'next_stage':'Fit amplitudes under hard direction constraints only after every direction passes.','limits':'Training support only. No infeasibility claim from unsuccessful optimization. Nonzero online moves, including small moves, require the correct sign; exact ties allow one point. Amplitude tolerance remains 15 percent for stage two.'}
    (D/('direction-first-fit.json' if args.scale=='online' else 'direction-first-fit-'+args.scale+'.json')).write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
