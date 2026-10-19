"""Rebuild an inequality separator from raw measured qualities in longdouble."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/distribution-fit'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--certificate',default='amplitude-global-510-certificate.json');args=parser.parse_args()
 path=D/args.certificate;p=json.loads(path.read_text());target=json.loads((D/'targets.json').read_text());edges=target['edges'];hashes={v['version']:v['source_sha256'] for v in target['versions']};n=len(edges)
 signed=np.zeros(n,dtype=np.longdouble);absolute=signed.copy();lower=np.longdouble(0)
 for c in p['constraints']:
  e=edges[c['edge']];sign=c['sign'];bound=sign*(e['online_total_delta']-e['linear_delta'])-e['tolerance']-1e-7 if c['kind']=='amplitude' else -sign*e['linear_delta']
  assert abs(bound-c['bound'])<1e-10 and c['coefficient']>=0
  weight=np.longdouble(c['coefficient']);signed[c['edge']]+=weight*sign;absolute[c['edge']]+=weight;lower+=weight*np.longdouble(bound)
 cache={};meta={};upper=np.longdouble('-inf');magnitude=np.longdouble(0)
 for label in p['population_ids']:
  rnd,atom,case=label.split('/');test=int(case.split('-')[0][4:]);values={}
  if rnd not in meta:meta[rnd]={a['id']:a for a in json.loads((ROOT/'local_distribution_fit'/rnd/'atoms.json').read_text())['atoms']}
  for v in {e[k] for e in edges for k in ('from','to')}:
   key=(rnd,atom,v)
   if key not in cache:
    cell=json.loads((D/rnd/'cells'/f'v{v}--{atom}.json').read_text());assert cell['complete'] and cell['source_sha256']==hashes[v] and cell['data_sha256']==meta[rnd][atom]['sha256'];cache[key]=cell
   values[v]=np.longdouble(next(x['quality'] for x in cache[key]['cases'] if x['test']==test))*25000
  a=np.array([values[e['to']]-values[e['from']] for e in edges],dtype=np.longdouble);upper=max(upper,signed@a);magnitude=max(magnitude,absolute@np.abs(a))
 guard=np.longdouble(1e-10)*(1+sum(abs(np.longdouble(c['coefficient'])*np.longdouble(c['bound'])) for c in p['constraints'])+magnitude);gap=lower-upper
 assert gap>guard
 out={'certificate_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'columns':len(p['population_ids']),'raw_verified_gap':float(gap),'numerical_guard':float(guard),'certified':True,'limits':f'Only these {len(p["population_ids"])} measured columns; unmeasured data can change feasibility.'}
 path.with_name(path.stem+'-verified.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
