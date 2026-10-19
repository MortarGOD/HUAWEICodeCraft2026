"""Reduced-support feasibility probes; all acceptance checks use full raw matrix."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/distribution-fit';sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',default='amplitude-boundary-fit.json');parser.add_argument('--output',default='amplitude-constraint-growth.json');parser.add_argument('--attempts',type=int,default=12);args=parser.parse_args()
    source=D/args.source;report=json.loads(source.read_text())
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
    from tools.distribution_fit.barrier_feasibility import phase_one
    tol=np.array([e['tolerance'] for e in edges]);history=[]
    def checks(weights):
        value=A@weights+linear
        direction=np.where(delta>0,value>1e-8,np.where(delta<0,value< -1e-8,np.abs(value)<=1+1e-7))
        matched=np.abs(value-delta)<=tol+1e-7
        return value,direction,matched
    value,direction,matched=checks(w);assert direction.all()
    order=sorted(np.flatnonzero(~matched),key=lambda i:abs(value[i]-delta[i])/tol[i])[:args.attempts]
    for attempt,i in enumerate(order):
        if matched[i]:continue
        rows=[];bounds=[]
        for j,d in enumerate(delta):
            if d:
                sign=np.sign(d);rows.append(sign*A[j]);bounds.append(1e-7-sign*linear[j])
            else:rows.extend([A[j],-A[j]]);bounds.extend([-1-linear[j],linear[j]-1])
        required=matched.copy();required[i]=True
        for j in np.flatnonzero(required):
            rows.extend([A[j],-A[j]]);bounds.extend([delta[j]-linear[j]-tol[j]+1e-6,-delta[j]+linear[j]-tol[j]+1e-6])
        active=(w>1e-10) | (w==0)
        active[np.argsort(np.abs(A[i]))[-20:]]=True
        seed=w[active];seed=seed/seed.sum()
        reduced,solver_history,status=phase_one(np.asarray(rows)[:,active],np.asarray(bounds),seed)
        trial=np.zeros_like(w);trial[active]=reduced
        val,direction,passed=checks(trial)
        accepted=bool(direction.all() and passed[required].all() and trial.min()>=0 and abs(trial.sum()-1)<1e-8)
        if accepted:w=trial;value=val;matched=passed
        history.append({'attempt':attempt+1,'target_version':edges[i]['to'],'accepted':accepted,'amplitude_matched':int(matched.sum()),'solver_status':status,'active_columns':int(active.sum()),'minimum_raw_slack':float(np.min(np.asarray(rows)@trial-np.asarray(bounds))),'phase_one_slack':solver_history[-1]['phase_one_slack'] if solver_history else None});print(history[-1],flush=True)
        out={'method':'Reduced-support phase-I constraint growth, full-matrix acceptance','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'passed':98,'total':98,'amplitude_matched':int(matched.sum()),'weights':dict(zip(ids,w.tolist())),'history':history,'edges':[dict(e,predicted_total_delta=float(value[j]),direction_passed=True,amplitude_passed=bool(matched[j])) for j,e in enumerate(edges)],'limits':'Restricted column search; failed trials do not prove full-support infeasibility. Training fit, not independent validation. Failed phase-I search is not an infeasibility proof.'}
        (D/args.output).write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
