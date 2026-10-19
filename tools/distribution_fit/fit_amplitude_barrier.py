"""Exact direction-boundary line searches on fully measured shared columns."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/distribution-fit';sys.path.insert(0,str(ROOT))

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
    rows=[];bounds=[]
    for i,d in enumerate(delta):
        if d:
            sign=np.sign(d);rows.append(sign*A[i]);bounds.append(1e-7-sign*linear[i])
        else:
            rows.extend([A[i],-A[i]]);bounds.extend([-1-linear[i],linear[i]-1])
    G=np.asarray(rows);h=np.asarray(bounds);norm=np.maximum(np.max(np.abs(G),axis=1),1.)
    G/=norm[:,None];h/=norm
    tol=np.array([e['tolerance'] for e in edges]);B=A/tol[:,None];y=(delta-linear)/tol
    def residual(z):
        d=B@z-y;return np.maximum(d-1,0)+np.minimum(d+1,0)
    def matches(z):return int((np.abs(A@z+linear-delta)<=tol+1e-7).sum())
    def barrier(z,mu):
        slack=G@z-h
        if z.min()<=0 or slack.min()<=0:return float('inf')
        r=residual(z);return float(.5*r@r-mu*(np.log(z).sum()+np.log(slack).sum()))
    assert np.min(w)>0 and np.min(G@w-h)>0
    best=w.copy();best_rank=(matches(w),-float(residual(w)@residual(w)));history=[];status='finished'
    for stage in range(14):
        mu=.1*(.2**stage)
        for iteration in range(80):
            slack=G@w-h;r=residual(w);active=np.abs(B@w-y)>1
            grad=B.T@r-mu*(1/w+G.T@(1/slack))
            H=B[active].T@B[active]+mu*(G.T@(G/slack[:,None]**2))
            H[np.diag_indices_from(H)]+=mu/w**2
            try:sol=np.linalg.solve(H,np.column_stack((-grad,np.ones(len(w)))))
            except np.linalg.LinAlgError:status='singular Newton system';break
            step=sol[:,0]-sol[:,1]*sol[:,0].sum()/sol[:,1].sum();descent=float(grad@step)
            if descent>=0:status='non-descent numerical step';break
            if -descent/2<max(mu*1e-6,1e-12):break
            alpha=1.;old=barrier(w,mu)
            while alpha>1e-16 and barrier(w+alpha*step,mu)>old+.01*alpha*descent:alpha*=.5
            if alpha<=1e-16:status='line search stalled';break
            w+=alpha*step
            rank=(matches(w),-float(residual(w)@residual(w)))
            if rank>best_rank:best=w.copy();best_rank=rank
        history.append({'stage':stage,'mu':mu,'amplitude_matched':matches(w),'normalized_interval_loss':float(residual(w)@residual(w)),'best_amplitude_matched':best_rank[0]});print(history[-1],flush=True)
    value=A@best+linear;direction=np.where(delta>0,value>0,np.where(delta<0,value<0,np.abs(value)<=1+1e-7));assert direction.all()
    out={'stage':'amplitude_with_hard_directions','method':'Interval-error Newton log-barrier with all direction inequalities hard constrained','status':status,'passed':int(direction.sum()),'total':len(edges),'amplitude_matched':best_rank[0],'weights':dict(zip(ids,best.tolist())),'history':history,'edges':[dict(e,predicted_total_delta=float(value[i]),direction_passed=bool(direction[i]),amplitude_passed=bool(abs(value[i]-delta[i])<=tol[i]+1e-7)) for i,e in enumerate(edges)],'parent_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'limits':'Current measured training support only. Best observed match count is not a global optimum. Independent raw-cell verification required.'}
    out['minimum_loss_weights']=dict(zip(ids,w.tolist()))
    out['minimum_loss_amplitude_matched']=matches(w)
    out['minimum_normalized_interval_loss']=float(residual(w)@residual(w))
    (D/'amplitude-hard-directions-fit.json').write_text(json.dumps(out,indent=2)+'\n');print('Final direction',out['passed'],'amplitude',out['amplitude_matched'],flush=True)
if __name__=='__main__':main()
