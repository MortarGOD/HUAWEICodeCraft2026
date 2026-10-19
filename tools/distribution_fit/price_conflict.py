"""Rank fully observed populations against a fixed reduced separating direction."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/distribution-fit'
def main():
 r=json.loads((D/'reduced-conflict-certificate.json').read_text());edges=r['constraints']
 current={(e['from'],e['to']):e for e in json.loads((D/'targets.json').read_text())['edges']}
 assert all((e['from'],e['to']) in current and all(e[k]==current[(e['from'],e['to'])][k] for k in ('target','tolerance')) for e in edges), 'Certificate targets are stale; rebuild for current acceptance before pricing.'
 c=np.array(r['support_separation']['separating_vector'],dtype=np.longdouble);out=[]
 lo=np.array([max(e['target']-e['tolerance'],1e-9) if e['target']>1 else e['target']-e['tolerance'] for e in edges],dtype=np.longdouble)
 hi=np.array([min(e['target']+e['tolerance'],-1e-9) if e['target']< -1 else e['target']+e['tolerance'] for e in edges],dtype=np.longdouble);required=float(np.sum(np.where(c>=0,c*lo,c*hi)))
 versions={e[k] for e in edges for k in ('from','to')}
 sources={x['version']:x['source_sha256'] for x in json.loads((D/'manifest.json').read_text())['rows']}
 for manifest in sorted((ROOT/'local_distribution_fit').glob('round*/atoms.json')):
  for atom in json.loads(manifest.read_text())['atoms']:
   if not atom['fit_eligible']:continue
   cells={}
   for v in versions:
    p=D/manifest.parent.name/'cells'/f'v{v}--{atom["id"]}.json'
    if not p.exists():break
    cell=json.loads(p.read_text())
    if not cell['complete'] or cell['source_sha256']!=sources[v] or cell['data_sha256']!=atom['sha256']:break
    cells[v]=cell
   if len(cells)!=len(versions):continue
   first=next(iter(cells.values()))['cases']
   assert all([(x['test'],x['tokens']) for x in cell['cases']]==[(x['test'],x['tokens']) for x in first] for cell in cells.values())
   for j,case in enumerate(first):
    vector=np.array([np.longdouble(cells[e['to']]['cases'][j]['quality'])-np.longdouble(cells[e['from']]['cases'][j]['quality']) for e in edges])*25000
    price=float(c@vector);out.append({'population':manifest.parent.name+'/'+atom['id'],'test':case['test'],'tokens':case['tokens'],'price':price,'required_lower_bound':required,'margin':price-required})
 out.sort(key=lambda x:-x['price']);report={'required_versions':sorted(versions),'ranked_cases':out,'limits':'Price against one fixed certificate only. A positive margin invalidates that certificate, not all infeasibility proofs and not proof of a feasible shared mixture.'}
 (D/'conflict-prices.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(out[:5]))
if __name__=='__main__':main()
