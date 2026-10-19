"""Rebuild an inequality separator from raw measured qualities in longdouble."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/distribution-fit'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--certificate',default='amplitude-global-510-certificate.json');parser.add_argument('--output',default='amplitude-reduced-510-certificate.json');args=parser.parse_args()
 path=D/args.certificate;p=json.loads(path.read_text());target=json.loads((D/'targets.json').read_text());edges=target['edges'];hashes={v['version']:v['source_sha256'] for v in target['versions']};n=len(edges)
 signed=np.zeros(n,dtype=np.longdouble);absolute=signed.copy();lower=np.longdouble(0)
 for c in p['constraints']:
  e=edges[c['edge']];sign=c['sign'];bound=sign*(e['online_total_delta']-e['linear_delta'])-e['tolerance']-1e-7 if c['kind']=='amplitude' else -sign*e['linear_delta']
  assert abs(bound-c['bound'])<1e-10 and c['coefficient']>=0
  weight=np.longdouble(c['coefficient']);signed[c['edge']]+=weight*sign;absolute[c['edge']]+=weight;lower+=weight*np.longdouble(bound)
 columns=[];cache={};meta={};upper=np.longdouble('-inf');magnitude=np.longdouble(0)
 for label in p['population_ids']:
  rnd,atom,case=label.split('/');test=int(case.split('-')[0][4:]);values={}
  if rnd not in meta:meta[rnd]={a['id']:a for a in json.loads((ROOT/'local_distribution_fit'/rnd/'atoms.json').read_text())['atoms']}
  for v in {e[k] for e in edges for k in ('from','to')}:
   key=(rnd,atom,v)
   if key not in cache:
    cell=json.loads((D/rnd/'cells'/f'v{v}--{atom}.json').read_text());assert cell['complete'] and cell['source_sha256']==hashes[v] and cell['data_sha256']==meta[rnd][atom]['sha256'];cache[key]=cell
   values[v]=np.longdouble(next(x['quality'] for x in cache[key]['cases'] if x['test']==test))*25000
  a=np.array([values[e['to']]-values[e['from']] for e in edges],dtype=np.longdouble);columns.append(a);upper=max(upper,signed@a);magnitude=max(magnitude,absolute@np.abs(a))
 A=np.array(columns,dtype=np.longdouble).T
 G=np.array([c['sign']*A[c['edge']] for c in p['constraints']]);h=np.array([c['bound'] for c in p['constraints']],dtype=np.longdouble);coeff=np.array([c['coefficient'] for c in p['constraints']],dtype=np.longdouble)
 def gap(c):
  lower=c@h;upper=np.max(c@G);guard=np.longdouble(1e-10)*(1+np.abs(c)@np.abs(h)+np.max(np.abs(c)@np.abs(G)))
  return lower-upper,guard
 assert gap(coeff)[0]>gap(coeff)[1]
 changed=True
 while changed:
  changed=False
  for i in np.argsort(coeff):
   if coeff[i]==0:continue
   trial=coeff.copy();trial[i]=0
   if gap(trial)[0]>gap(trial)[1]:coeff=trial;changed=True
 g,guard=gap(coeff);p['constraints']=[dict(c,coefficient=float(coeff[i])) for i,c in enumerate(p['constraints']) if coeff[i]>0]
 p.update(gap=float(g),numerical_guard=float(guard),certified=True,reduction_parent_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),limits='Greedily reduced verified separator; not a minimum conflict. Current measured support only.')
 (D/args.output).write_text(json.dumps(p,indent=2)+'\n')
 vs={edges[c['edge']][k] for c in p['constraints'] for k in ('from','to')}
 print('constraints',len(p['constraints']),'versions',len(vs),'gap',float(g),'guard',float(guard))
if __name__=='__main__':main()
