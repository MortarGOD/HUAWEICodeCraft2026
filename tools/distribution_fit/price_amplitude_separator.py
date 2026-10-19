"""Price measured candidates against the current global amplitude separator."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/distribution-fit'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--certificate',default='amplitude-global-510-certificate.json');args=parser.parse_args()
 path=D/args.certificate;p=json.loads(path.read_text());verification=json.loads(path.with_name(path.stem+'-verified.json').read_text());assert verification['certified'] and verification['certificate_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
 edges=json.loads((D/'targets.json').read_text())['edges'];signed=np.zeros(len(edges),dtype=np.longdouble);lower=np.longdouble(0)
 for c in p['constraints']:
  e=edges[c['edge']];bound=c['sign']*(e['online_total_delta']-e['linear_delta'])-e['tolerance']-1e-7 if c['kind']=='amplitude' else -c['sign']*e['linear_delta'];assert abs(bound-c['bound'])<1e-10
  signed[c['edge']]+=np.longdouble(c['coefficient'])*c['sign'];lower+=np.longdouble(c['coefficient'])*bound
 active=np.flatnonzero(signed);versions=sorted({edges[i][k] for i in active for k in ('from','to')});hashes={v['version']:v['source_sha256'] for v in json.loads((D/'targets.json').read_text())['versions']};out=[];missing=[]
 for m in sorted((ROOT/'local_distribution_fit').glob('round*/atoms.json')):
  for a in json.loads(m.read_text())['atoms']:
   if not a['fit_eligible']:continue
   cells={};need=[]
   for v in versions:
    f=D/m.parent.name/'cells'/f'v{v}--{a["id"]}.json'
    if not f.exists():need.append(v);continue
    c=json.loads(f.read_text())
    if not c['complete'] or c['source_sha256']!=hashes[v] or c['data_sha256']!=a['sha256']:need.append(v);continue
    cells[v]={x['test']:x for x in c['cases']}
   population=m.parent.name+'/'+a['id']
   if need:missing.append({'population':population,'missing_versions':need});continue
   for j in next(iter(cells.values())):
    value=sum(signed[i]*(np.longdouble(cells[edges[i]['to']][j]['quality'])-np.longdouble(cells[edges[i]['from']][j]['quality']))*25000 for i in active)
    label=population+f'/test{j}-tokens{next(iter(cells.values()))[j]["tokens"]}'
    out.append({'population':population,'test':j,'margin':float(value-lower),'in_complete_support':label in p['population_ids']})
 out.sort(key=lambda r:-r['margin']);r={'certificate_sha256':verification['certificate_sha256'],'required_versions':versions,'ranked_cases':out,'incomplete':missing,'limits':'Positive margin breaks this separator only, not proof of joint fitting. Original targets and all-direction requirements remain.'}
 (D/'amplitude-separator-prices.json').write_text(json.dumps(r,indent=2)+'\n');print('required versions',len(versions));print(json.dumps(out[:5]));print('closest incomplete',sorted(missing,key=lambda x:len(x['missing_versions']))[:5])
if __name__=='__main__':main()
