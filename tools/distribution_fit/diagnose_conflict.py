"""Shrink a separating vector without changing targets or fitting acceptance."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/distribution-fit'
def main():
 r=json.loads((D/'case-fit.json').read_text());assert r['support_separation']['certified']
 edges=r['edges'];cache={};columns=[]
 for label in r['population_ids']:
  population,testpart=label.rsplit('/test',1);rnd,atom=population.split('/',1);test,tokens=map(int,testpart.split('-tokens'));values={}
  for v in {e[k] for e in edges for k in ('from','to')}:
   key=(rnd,atom,v)
   if key not in cache:cache[key]=json.loads((D/rnd/'cells'/f'v{v}--{atom}.json').read_text())
   cell=cache[key];assert cell['complete'];item=next(x for x in cell['cases'] if x['test']==test);assert item['tokens']==tokens
   values[v]=np.longdouble(item['quality'])*25000
  columns.append([values[e['to']]-values[e['from']] for e in edges])
 A=np.array(columns,dtype=np.longdouble).T;c=np.array(r['support_separation']['separating_vector'],dtype=np.longdouble)
 lo=np.array([e['target']-e['tolerance'] for e in edges],dtype=np.longdouble);hi=np.array([e['target']+e['tolerance'] for e in edges],dtype=np.longdouble)
 for i,e in enumerate(edges):
  if e['target']>1:lo[i]=max(lo[i],1e-9)
  if e['target']< -1:hi[i]=min(hi[i],-1e-9)
 def separation(vector):
  lower=np.sum(np.where(vector>=0,vector*lo,vector*hi));upper=np.max(vector@A)
  guard=1e-10*(1+np.sum(np.abs(vector)*np.maximum(np.abs(lo),np.abs(hi)))+np.max(np.abs(vector)@np.abs(A)))
  return lower-upper,guard
 assert separation(c)[0]>separation(c)[1]
 changed=True
 while changed:
  changed=False
  for i in np.argsort(np.abs(c)):
   if c[i]==0:continue
   candidate=c.copy();candidate[i]=0;gap,guard=separation(candidate)
   if gap>guard:c=candidate;changed=True
 gap,guard=separation(c);active=np.flatnonzero(c)
 report={'columns':len(columns),'population_ids':r['population_ids'],'constraints':[dict(edges[i],separating_coefficient=float(c[i])) for i in active],'gap':float(gap),'numerical_guard':float(guard),'limits':'Greedy reduced separating vector for current complete support; not a minimum infeasible subsystem or a claim about unmeasured populations.'}
 (D/'reduced-conflict.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'columns':len(columns),'constraints':[(edges[i]['research_parent'],edges[i]['to']) for i in active],'gap':float(gap)}))
if __name__=='__main__':main()
