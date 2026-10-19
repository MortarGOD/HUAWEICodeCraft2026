"""Rebuild an inequality separator with 80-digit Decimal arithmetic."""
import argparse,json,hashlib
from pathlib import Path
from decimal import Decimal, getcontext
getcontext().prec=80
def dec(x):
 return Decimal.from_float(x) if isinstance(x,float) else Decimal(x)
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/distribution-fit'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--certificate',default='amplitude-global-510-certificate.json');args=parser.parse_args()
 path=D/args.certificate;p=json.loads(path.read_text());target_path=D/p.get('targets_file','targets.json');target=json.loads(target_path.read_text());edges=target['edges'];hashes={v['version']:v['source_sha256'] for v in target['versions']};n=len(edges)
 if p.get('targets_sha256'):assert hashlib.sha256(target_path.read_bytes()).hexdigest()==p['targets_sha256']
 signed=[Decimal(0) for _ in range(n)];absolute=signed.copy();lower=Decimal(0)
 for c in p['constraints']:
  e=edges[c['edge']];sign=int(c['sign']);assert sign==c['sign'] and sign in (-1,1);bound=sign*(dec(e['online_total_delta'])-dec(e['linear_delta']))-dec(e['tolerance'])-Decimal('0.0000001') if c['kind']=='amplitude' else -sign*dec(e['linear_delta'])
  assert c['kind'] in ('amplitude','direction','tie')
  if c['kind']=='tie':
   assert e['online_total_delta']==0
   bound=-Decimal(1)-Decimal('0.0000001')-sign*dec(e['linear_delta'])
  assert abs(bound-dec(c['bound']))<Decimal('1e-10') and c['coefficient']>=0
  weight=dec(c['coefficient']);signed[c['edge']]+=weight*sign;absolute[c['edge']]+=weight;lower+=weight*dec(bound)
 cache={};meta={};upper=dec('-inf');magnitude=dec(0)
 for label in p['population_ids']:
  rnd,atom,case=label.split('/');test=int(case.split('-')[0][4:]);values={}
  if rnd not in meta:meta[rnd]={a['id']:a for a in json.loads((ROOT/'local_distribution_fit'/rnd/'atoms.json').read_text())['atoms']}
  for v in {e[k] for e in edges for k in ('from','to')}:
   key=(rnd,atom,v)
   if key not in cache:
    cell=json.loads((D/rnd/'cells'/f'v{v}--{atom}.json').read_text());assert cell['complete'] and cell['source_sha256']==hashes[v] and cell['data_sha256']==meta[rnd][atom]['sha256'];cache[key]=cell
   values[v]=dec(next(x['quality'] for x in cache[key]['cases'] if x['test']==test))*25000
  a=[values[e['to']]-values[e['from']] for e in edges];upper=max(upper,sum(x*y for x,y in zip(signed,a)));magnitude=max(magnitude,sum(x*abs(y) for x,y in zip(absolute,a)))
 guard=dec(1e-10)*(1+sum(abs(dec(c['coefficient'])*dec(c['bound'])) for c in p['constraints'])+magnitude);gap=lower-upper
 assert gap>guard
 out={'certificate_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'columns':len(p['population_ids']),'arithmetic':'80-digit Decimal of stored binary64 measurements','decimal_gap':str(gap),'raw_verified_gap':float(gap),'numerical_guard':float(guard),'certified':True,'limits':f'Only these {len(p["population_ids"])} measured columns; unmeasured data can change feasibility.'}
 path.with_name(path.stem+'-decimal-verified.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
