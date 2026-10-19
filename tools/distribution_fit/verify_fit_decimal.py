"""Independently audit stored binary64 observations with 80-digit Decimal sums."""
import argparse,hashlib,json
from pathlib import Path
from decimal import Decimal, getcontext
getcontext().prec=80
def dec(x):
 return Decimal.from_float(x) if isinstance(x,float) else Decimal(x)
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/distribution-fit'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('fit');args=parser.parse_args()
 path=D/args.fit;p=json.loads(path.read_text());target_path=D/p.get('targets_file','targets.json');target=json.loads(target_path.read_text())
 if 'targets_sha256' in p:assert hashlib.sha256(target_path.read_bytes()).hexdigest()==p['targets_sha256']
 edges=target['edges'];assert [(e['from'],e['to']) for e in edges]==[(e['from'],e['to']) for e in p['edges']]
 weights=p['weights'];assert min(weights.values())>=-1e-15 and abs(sum(weights.values())-1)<1e-10
 for group in p.get('seed_groups',[]):
  values=[dec(weights[label]) for label in group['members']]
  assert len(values)==3 and all(value==values[0] for value in values), 'Unequal seed-group weights'
 hashes={r['version']:r['source_sha256'] for r in target['versions']};scores={};cache={};meta={}
 for v in {e[k] for e in edges for k in ('from','to')}:
  s=dec(0)
  for label,w in weights.items():
   if not w:continue
   rnd,atom,case=label.split('/');j=int(case.split('-')[0][4:]);key=(rnd,atom,v)
   if rnd not in meta:meta[rnd]={a['id']:a for a in json.loads((ROOT/'local_distribution_fit'/rnd/'atoms.json').read_text())['atoms']}
   if key not in cache:
    c=json.loads((D/rnd/'cells'/f'v{v}--{atom}.json').read_text())
    assert c['complete'] and c['source_sha256']==hashes[v] and c['data_sha256']==meta[rnd][atom]['sha256']
    cache[key]=c
   q=next(x['quality'] for x in cache[key]['cases'] if x['test']==j)
   s+=dec(w)*dec(q)*25000
  scores[v]=s
 detail=[]
 for e,claimed in zip(edges,p['edges']):
  value=scores[e['to']]-scores[e['from']]+dec(e['linear_delta']);d=e['online_total_delta']
  ok=value>0 if d>0 else value<0 if d<0 else abs(value)<=Decimal('1.0000001')
  detail.append({'from':e['from'],'to':e['to'],'predicted_total_delta':float(value),'passed':bool(ok),'agrees_with_claim':bool(ok)==claimed['direction_passed']})
 if any('amplitude_passed' in e for e in p['edges']):
  for actual,e,claimed in zip(detail,edges,p['edges']):
   value=scores[e['to']]-scores[e['from']]+dec(e['linear_delta'])
   actual['decimal_predicted_total_delta']=str(value)
   actual['amplitude_passed']=abs(value-dec(e['online_total_delta']))<=dec(e['tolerance'])+Decimal('0.0000001')
   actual['amplitude_claim_agrees']=actual['amplitude_passed']==claimed['amplitude_passed']
 out={'fit_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'passed':sum(x['passed'] for x in detail),'total':len(detail),'all_claims_agree':all(x['agrees_with_claim'] for x in detail),'edges':detail,'method':'Independent 80-digit Decimal weighted raw quality sums from stored binary64 values; parent subtraction after summation; current measured Linear corrections and target directions; cell source/data identities checked.'}
 if any('amplitude_passed' in e for e in detail):
  out['amplitude_matched']=sum(e['amplitude_passed'] for e in detail)
  out['all_amplitude_claims_agree']=all(e['amplitude_claim_agrees'] for e in detail)
  assert out['all_amplitude_claims_agree']
 dest=path.with_name(path.stem+'-decimal-verified.json');dest.write_text(json.dumps(out,indent=2)+'\n')
 print(out['passed'],out['total'],out['all_claims_agree'])
 assert out['all_claims_agree']
if __name__=='__main__':main()
