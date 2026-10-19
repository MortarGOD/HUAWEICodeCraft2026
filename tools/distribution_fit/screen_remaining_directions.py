"""Screen new measured columns for repairing current failed direction subset."""
import argparse,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/distribution-fit'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--fit',default='direction-first-fit.json');args=parser.parse_args()
 p=json.loads((D/args.fit).read_text());edges=[e for e in p['edges'] if not e['direction_passed']]
 versions={e[k] for e in edges for k in ('from','to')};sources={v['version']:v['source_sha256'] for v in json.loads((D/'targets.json').read_text())['versions']}
 out=[]
 for manifest in sorted((ROOT/'local_distribution_fit').glob('round*/atoms.json')):
  rnd=manifest.parent.name
  for a in json.loads(manifest.read_text())['atoms']:
   if not a['fit_eligible']:continue
   cells={}
   for v in versions:
    f=D/rnd/'cells'/f'v{v}--{a["id"]}.json'
    if not f.exists():break
    c=json.loads(f.read_text())
    if not c['complete'] or c['source_sha256']!=sources[v] or c['data_sha256']!=a['sha256']:break
    cells[v]={x['test']:x for x in c['cases']}
   if len(cells)!=len(versions):continue
   for j in next(iter(cells.values())):
    lo=0.;hi=1.;detail=[]
    for e in edges:
     assert e['online_total_delta']!=0
     end=(cells[e['to']][j]['quality']-cells[e['from']][j]['quality'])*25000+e['linear_delta']
     base=e['predicted_total_delta'];sign=np.sign(e['online_total_delta']);slope=sign*(end-base);offset=sign*base
     if slope>0:lo=max(lo,(1e-7-offset)/slope)
     elif slope<0:hi=min(hi,(1e-7-offset)/slope)
     elif offset<1e-7:hi=-1
     detail.append({'to':e['to'],'candidate_total_delta':end})
    if 0<=lo<hi<=1:
     label=rnd+'/'+a['id']+f'/test{j}-tokens{next(iter(cells.values()))[j]["tokens"]}'
     out.append({'population':rnd+'/'+a['id'],'case_id':label,'test':j,'minimum_mixture_fraction':lo,'maximum_mixture_fraction':hi,'already_in_complete_support':label in p['weights'],'predictions':detail})
 out.sort(key=lambda x:x['minimum_mixture_fraction'])
 report={'base_passed':p.get('passed',p.get('verified_longdouble_passed')),'source_fit':args.fit,'failed_edges':[e['to'] for e in edges],'required_versions':sorted(versions),'ranked_candidates':out,'limits':'Only currently failed directions screened. Other directions may regress. Full measured support and all-98 joint check required before promotion.'}
 (D/'remaining-direction-candidates.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps([x for x in out if not x['already_in_complete_support']][:5]))
if __name__=='__main__':main()
