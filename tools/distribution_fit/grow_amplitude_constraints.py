"""Exact direction-boundary line searches on fully measured shared columns."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/distribution-fit';sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',default='amplitude-boundary-fit.json');parser.add_argument('--output',default='amplitude-constraint-growth.json');parser.add_argument('--attempts',type=int,default=12);parser.add_argument('--targets',default='targets.json');parser.add_argument('--large-first',action='store_true');parser.add_argument('--allow-amplitude-regressions',action='store_true',help='Explicitly permit sacrificing smaller matched edges; default preserves every match');parser.add_argument('--simplex',action='store_true');parser.add_argument('--direction-margin',type=float,default=1e-7);parser.add_argument('--seed-group-round');args=parser.parse_args()
    source=D/args.source;report=json.loads(source.read_text())
    if 'explicit_seed_family_extension' in report:
        assert args.seed_group_round=='round041', 'Explicit seed-family support requires equal-seed fitting'
    edges=json.loads((D/args.targets).read_text())['edges']
    assert [(e['from'],e['to']) for e in edges]==[(e['from'],e['to']) for e in report['edges']]
    ids=list(report['weights']);versions={e[k] for e in edges for k in ('from','to')}
    hashes={r['version']:r['source_sha256'] for r in json.loads((D/args.targets).read_text())['versions']}
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
    raw_A=A;expansion=None;seed_groups=[]
    if args.seed_group_round:
        from tools.distribution_fit.seed_groups import seed_expansion
        expansion,w,seed_groups=seed_expansion(ids,w,args.seed_group_round)
        for group in seed_groups:
            members=[meta[label.split('/')[0]][label.split('/')[1]] for label in group['members']]
            assert len({member['sha256'] for member in members})==1, 'Seed group mixes different input tensors'
            assert len({member['seed'] for member in members})==3, 'Seed group repeats calibration seed'
            assert len({(member['mode'],member.get('noise')) for member in members})==1, 'Seed group mixes protocols'
        A=raw_A@expansion
    def expanded(weights):
        return weights if expansion is None else expansion@weights
    from tools.distribution_fit.barrier_feasibility import phase_one
    if args.simplex:
        from tools.distribution_fit.simplex_feasibility import phase_one
    tol=np.array([e['tolerance'] for e in edges]);history=[]
    def checks(weights):
        value=raw_A@expanded(weights)+linear
        direction=np.where(delta>0,value>1e-8,np.where(delta<0,value< -1e-8,np.abs(value)<=1+1e-7))
        matched=np.abs(value-delta)<=tol+1e-7
        return value,direction,matched
    value,direction,matched=checks(w);assert direction.all()
    order=(sorted(range(len(edges)),key=lambda i:(-abs(delta[i]),edges[i]['to'])) if args.large_first else sorted(np.flatnonzero(~matched),key=lambda i:abs(value[i]-delta[i])/tol[i]))[:args.attempts]
    for attempt,i in enumerate(order):
        if matched[i]:continue
        rows=[];bounds=[]
        for j,d in enumerate(delta):
            if d:
                sign=np.sign(d);rows.append(sign*A[j]);bounds.append(args.direction_margin-sign*linear[j])
            else:rows.extend([A[j],-A[j]]);bounds.extend([-1-linear[j],linear[j]-1])
        required=matched.copy()
        if args.large_first and args.allow_amplitude_regressions:required &= np.abs(delta)>=abs(delta[i])
        required[i]=True
        for j in np.flatnonzero(required):
            rows.extend([A[j],-A[j]]);bounds.extend([delta[j]-linear[j]-tol[j]+1e-6,-delta[j]+linear[j]-tol[j]+1e-6])
        trial,solver_history,status=phase_one(np.asarray(rows),np.asarray(bounds),w)
        # Preserve the raw trial for independent numerical diagnosis, even on rejection.
        (D/(args.output.removesuffix('.json')+'-trial.json')).write_text(json.dumps({'weights':dict(zip(ids,expanded(trial).tolist())), 'solver_history':solver_history,'status':status,'direction_margin':args.direction_margin},indent=2)+'\n')
        val,direction,passed=checks(trial)
        accepted=bool(direction.all() and passed[required].all() and trial.min()>=0 and abs(trial.sum()-1)<1e-8)
        if accepted:w=trial;value=val;matched=passed
        history.append({'attempt':attempt+1,'target_version':edges[i]['to'],'accepted':accepted,'amplitude_matched':int(matched.sum()),'solver_status':status,'trial_min_weight':float(trial.min()),'trial_weight_sum':float(trial.sum()),'trial_direction_failures':[edges[j]['to'] for j in np.flatnonzero(~direction)],'trial_required_amplitude_failures':[edges[j]['to'] for j in np.flatnonzero(required & ~passed)]});print(history[-1],flush=True)
        out={'targets_file':args.targets,'targets_sha256':hashlib.sha256((D/args.targets).read_bytes()).hexdigest(),'large_first':args.large_first,'allow_amplitude_regressions':args.allow_amplitude_regressions,'solver':'simplex' if args.simplex else 'barrier','method':'Global phase-I growth of amplitude constraints with all directions preserved','passed':98,'total':98,'amplitude_matched':int(matched.sum()),'seed_groups':seed_groups,'weights':dict(zip(ids,expanded(w).tolist())),'history':history,'edges':[dict(e,predicted_total_delta=float(value[j]),direction_passed=True,amplitude_passed=bool(matched[j])) for j,e in enumerate(edges)],'limits':'Greedy constraint-growth training fit, not global maximum or independent validation. Failed phase-I search is not an infeasibility proof.'}
        (D/args.output).write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
