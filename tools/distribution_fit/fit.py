"""One shared nonnegative population mixture; explicit interval feasibility audit."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
def simplex(x):
 u=np.sort(x)[::-1];cs=np.cumsum(u)-1
 rho=np.flatnonzero(u-cs/np.arange(1,len(u)+1)>0)[-1]
 return np.maximum(x-cs[rho]/(rho+1),0)

def solve(A,y,tol,low,high,initial=None):
 B=A/tol[:,None];b=y/tol;n=A.shape[1]
 lo=low/tol;hi=high/tol
 w=np.full(n,1/n) if initial is None else simplex(initial);z=w.copy();t=1.;L=max(float(np.linalg.norm(B,2)**2),1e-30)
 best=w.copy();best_loss=float("inf")
 for iteration in range(100000):
  value=B@z
  residual=np.maximum(value-hi,0)+np.minimum(value-lo,0)
  nxt=simplex(z-B.T@residual/L)
  nv=B@nxt;loss=float(np.sum((np.maximum(nv-hi,0)+np.minimum(nv-lo,0))**2))
  if loss<best_loss:best=nxt.copy();best_loss=loss
  nt=(1+np.sqrt(1+4*t*t))/2
  if np.max(np.abs(nxt-w))<1e-12 and iteration>100:break
  z=nxt+(t-1)/nt*(nxt-w);w=nxt;t=nt
 return best,iteration+1
def separation_certificate(A,low,high,w):
 # Any c gives a necessary inequality min(c @ interval) <= max(c @ columns).
 # A strict reversed inequality excludes every simplex mixture of these columns.
 dtype=np.longdouble
 matrix=np.asarray(A,dtype=dtype);lo=np.asarray(low,dtype=dtype);hi=np.asarray(high,dtype=dtype)
 value=matrix@np.asarray(w,dtype=dtype)
 tolerance=np.maximum((hi-lo)/2,dtype(1e-12))
 c=(np.clip(value,lo,hi)-value)/tolerance**2
 norm=np.max(np.abs(c))
 if norm==0:return {'certified':False}
 c/=norm
 box_min=np.sum(np.where(c>=0,c*lo,c*hi));support_max=np.max(c@matrix)
 gap=box_min-support_max
 guard=dtype(1e-10)*(1+np.sum(np.abs(c)*np.maximum(np.abs(lo),np.abs(hi)))+np.max(np.abs(c)@np.abs(matrix)))
 return {'certified':bool(gap>guard),'separating_vector':c.astype(float).tolist(),
  'minimum_over_required_intervals':float(box_min),'maximum_over_available_columns':float(support_max),
  'gap':float(gap),'numerical_guard':float(guard),
  'scope':'Only the currently measured complete columns; no claim about unmeasured populations.'}

ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/distribution-fit';sys.path.insert(0,str(ROOT))
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--case-mixture',action='store_true');parser.add_argument('--solver',choices=['fista','admm'],default='fista');args=parser.parse_args()
 targets=json.loads((D/'targets.json').read_text());versions=[r['version'] for r in targets['versions']]
 atoms=[];columns=[];ids=[]
 for manifest in sorted((ROOT/'local_distribution_fit').glob('round*/atoms.json')):
  atoms.extend(dict(a,round=manifest.parent.name) for a in json.loads(manifest.read_text())['atoms'])
 for atom in atoms:
  if not atom['fit_eligible']:continue
  values=[]
  for v in versions:
   path=D/atom['round']/f'cells/v{v}--{atom["id"]}.json'
   if not path.exists():break
   r=json.loads(path.read_text())
   if not r['complete'] or r['source_sha256']!=next(x['source_sha256'] for x in targets['versions'] if x['version']==v) or r['data_sha256']!=atom['sha256']:break
   values.append(r['cases'] if args.case_mixture else r['mean_quality'])
  if len(values)==len(versions):
   label=atom['round']+'/'+atom['id']
   if args.case_mixture:
    signature=[(c['test'],c['tokens']) for c in values[0]]
    assert all([(c['test'],c['tokens']) for c in row]==signature for row in values)
    for j,(test,tokens) in enumerate(signature):
     columns.append([row[j]['quality'] for row in values]);ids.append(label+f'/test{test}-tokens{tokens}')
   else:columns.append(values);ids.append(label)
 report={'complete':False,'evaluated_versions':len(versions),'complete_population_columns':len(ids),'population_ids':ids,'all_matched':False,'unresolved_linear_anchor_versions':targets['unresolved'],'unavailable_parent_receipts':targets.get('unavailable_parent_receipts',[])}
 if columns:
  values=np.asarray(columns).T*25000;index={v:i for i,v in enumerate(versions)};edges=targets['edges']
  A=np.asarray([values[index[e['to']]]-values[index[e['from']]] for e in edges]);y=np.array([e['target'] for e in edges]);tol=np.array([e['tolerance'] for e in edges]);low=y-tol;high=y+tol
  low=np.where(y>1,np.maximum(low,1e-9),low);high=np.where(y< -1,np.minimum(high,-1e-9),high)
  initial=None
  prior=D/'round001/fit.json'
  if args.case_mixture and prior.exists():
   weights=json.loads(prior.read_text()).get('weights',{});counts={}
   for label in ids:counts[label.rsplit('/test',1)[0]]=counts.get(label.rsplit('/test',1)[0],0)+1
   initial=np.array([weights.get(label.rsplit('/test',1)[0],0)/counts[label.rsplit('/test',1)[0]] for label in ids])
   if initial.sum()==0:initial=None
  w,iterations=solve(A,y,tol,low,high,initial);report['solver']={'method':'simplex FISTA interval feasibility','iterations':iterations,'limitations':'No LP infeasibility certificate; failed fit does not prove no mixture exists.'}
  if args.solver=='admm':
   from tools.distribution_fit.feasibility import solve_feasible
   w,report['solver']=solve_feasible(A,low,high,w)
  report['support_separation']=separation_certificate(A,low,high,w)
  pred=A@w;detail=[]
  for i,e in enumerate(edges):
   passed=low[i]-1e-7<=pred[i]<=high[i]+1e-7
   detail.append(dict(e,predicted=float(pred[i]),residual=float(pred[i]-y[i]),passed=bool(passed),support_min=float(A[i].min()),support_max=float(A[i].max())))
  report.update(complete=True,weights=dict(zip(ids,w.tolist())),matched_edges=sum(x['passed'] for x in detail),total_edges=len(detail),edges=detail,feasible_weights_found=all(x['passed'] for x in detail),all_matched=all(x['passed'] for x in detail) and not targets['unresolved'],limits='Training support fit, no independent seed validation yet. Paired protocol not fitted. No per-version score correction. Missing effective parent receipts are separately marked. All matched refers to assessable research-parent constraints; independent validation remains outstanding.')
 report['mixture_unit']='individual test case with original calibration' if args.case_mixture else 'equal-weight test group'
 dest=D/('case-fit.json' if args.case_mixture else 'round001/fit.json');dest=dest.with_name(dest.stem+'-admm.json') if args.solver=='admm' else dest;dest.parent.mkdir(exist_ok=True);temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(dest);print(json.dumps({k:v for k,v in report.items() if k not in ('edges','weights')},ensure_ascii=False))
if __name__=='__main__':main()
