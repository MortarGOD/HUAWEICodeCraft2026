"""Exact direction-boundary line searches on fully measured shared columns."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/distribution-fit';sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',default='amplitude-growth-510.json');parser.add_argument('--output',default='amplitude-global-510-certificate.json');parser.add_argument('--target-version',type=int);parser.add_argument('--simplex',action='store_true');args=parser.parse_args()
    source=D/args.source;report=json.loads(source.read_text())
    targets_path=D/report.get('targets_file','targets.json')
    targets=json.loads(targets_path.read_text())
    if report.get('targets_sha256'):assert hashlib.sha256(targets_path.read_bytes()).hexdigest()==report['targets_sha256']
    edges=targets['edges']
    assert [(e['from'],e['to']) for e in edges]==[(e['from'],e['to']) for e in report['edges']]
    ids=list(report['weights']);versions={e[k] for e in edges for k in ('from','to')}
    hashes={r['version']:r['source_sha256'] for r in targets['versions']}
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
    from tools.distribution_fit.barrier_feasibility import phase_one
    if args.simplex:
        from tools.distribution_fit.simplex_feasibility import phase_one
    required=np.ones(len(edges),dtype=bool) if args.target_version is None else np.array([e['amplitude_passed'] or e['to']==args.target_version for e in report['edges']]); assert args.target_version is None or any(e['to']==args.target_version for e in edges)
    rows=[];bounds=[];labels=[]
    for i,e in enumerate(edges):
        for sign in ((1,-1) if required[i] else ()):
            rows.append(sign*A[i]);bounds.append(sign*(delta[i]-linear[i])-e['tolerance']-1e-7);labels.append({'edge':i,'kind':'amplitude','sign':sign})
        if delta[i]:
            sign=float(np.sign(delta[i]));rows.append(sign*A[i]);bounds.append(-sign*linear[i]);labels.append({'edge':i,'kind':'direction','sign':sign})
        else:
            for sign in (1,-1):
                rows.append(sign*A[i]);bounds.append(-1-1e-7-sign*linear[i]);labels.append({'edge':i,'kind':'tie','sign':sign})
    G=np.asarray(rows);h=np.asarray(bounds);weights,history,status=phase_one(G,h,w)
    (D/(args.output.removesuffix('.json')+'-phase-one-weights.json')).write_text(json.dumps({'weights':dict(zip(ids,weights.tolist())),'status':status,'history':history},indent=2)+'\n')
    phase_scale=np.maximum(np.max(np.abs(G),axis=1),1.)
    phase_slack=G@weights/phase_scale+history[-1]['phase_one_slack']-h/phase_scale
    barrier_dual=None if np.min(phase_slack)<=0 else 1/phase_slack/phase_scale
    from tools.distribution_fit.fit import solve
    y=delta-linear;tol=np.array([e['tolerance']+1e-7 for e in edges]);low=np.where(required,y-tol,-np.inf);high=np.where(required,y+tol,np.inf)
    low=np.where(delta>0,np.maximum(low,-linear),low);high=np.where(delta<0,np.minimum(high,-linear),high)
    low=np.where(delta==0,np.maximum(low,-1-1e-7-linear),low);high=np.where(delta==0,np.minimum(high,1+1e-7-linear),high)
    for refinement in range(8):
        weights,_=solve(A,y,tol,low,high,weights)
    signed=(np.clip(A@weights,low,high)-A@weights)/tol**2
    fallback=np.zeros(len(h))
    for i,coefficient in enumerate(signed):
        if coefficient==0:continue
        options=[j for j,label in enumerate(labels) if label['edge']==i and label['sign']==np.sign(coefficient)]
        j=max(options,key=lambda j:h[j]);fallback[j]=abs(coefficient)
    scale=np.maximum(np.max(np.abs(G),axis=1),1.);t=history[-1]['phase_one_slack']
    slack=G@weights/scale+t-h/scale
    dual=fallback;dual/=dual.max()
    candidates=[('fista',dual)]
    if args.simplex:
        simplex_dual=np.asarray(history[-1]['dual']);
        if simplex_dual.max()>0:candidates.append(('simplex',simplex_dual/simplex_dual.max()))
    if barrier_dual is not None:candidates.append(('barrier',barrier_dual/barrier_dual.max()))
    def margin(item):
        c=item[1].astype(np.longdouble);matrix=G.astype(np.longdouble);rhs=h.astype(np.longdouble)
        gap=c@rhs-np.max(c@matrix)
        guard=np.longdouble(1e-10)*(1+np.abs(c)@np.abs(rhs)+np.max(np.abs(c)@np.abs(matrix)))
        return float(gap-guard)
    dual_method,dual=max(candidates,key=margin)
    matrix=G.astype(np.longdouble);rhs=h.astype(np.longdouble);c=dual.astype(np.longdouble)
    lower=c@rhs;upper=np.max(c@matrix);gap=lower-upper
    guard=np.longdouble(1e-10)*(1+np.abs(c)@np.abs(rhs)+np.max(np.abs(c)@np.abs(matrix)))
    out={'targets_file':targets_path.name,'targets_sha256':hashlib.sha256(targets_path.read_bytes()).hexdigest(),'method':'Global phase-I and nonnegative inequality separator','dual_method':dual_method,'target_version':args.target_version,'required_amplitude_versions':[e['to'] for i,e in enumerate(edges) if required[i]],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'status':status,'columns':len(ids),'population_ids':ids,'edges':edges,'constraints':[dict(label,bound=float(h[i]),coefficient=float(c[i])) for i,label in enumerate(labels)],'gap':float(gap),'numerical_guard':float(guard),'certified':bool(gap>guard),'history':history,'limits':'A positive verified gap excludes joint amplitude feasibility only on these measured columns, using weaker non-strict direction bounds. It says nothing about unmeasured populations. Numerical solver failure alone is not proof.'}
    (D/args.output).write_text(json.dumps(out,indent=2)+'\n')
    print({k:v for k,v in out.items() if k not in ('population_ids','edges','constraints','history')},flush=True)
if __name__=='__main__':main()
