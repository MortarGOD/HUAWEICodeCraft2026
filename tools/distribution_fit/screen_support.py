"""Audit per-edge support using available cells; never label partial matrices fitted."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/distribution-fit'
def main():
 targets=json.loads((D/'targets.json').read_text());rows={r['version']:r for r in targets['versions']};result=[]
 for edge in targets['edges']:
  points=[]
  for manifest in sorted((ROOT/'local_distribution_fit').glob('round*/atoms.json')):
   for atom in json.loads(manifest.read_text())['atoms']:
    if not atom['fit_eligible']:continue
    cells=[]
    for version in (edge['from'],edge['to']):
     p=D/manifest.parent.name/f'cells/v{version}--{atom["id"]}.json'
     if not p.exists():break
     cell=json.loads(p.read_text())
     if not cell['complete'] or cell['source_sha256']!=rows[version]['source_sha256'] or cell['data_sha256']!=atom['sha256']:break
     cells.append(cell)
    if len(cells)!=2:continue
    a,b=cells
    assert [(c['test'],c['tokens']) for c in a['cases']]==[(c['test'],c['tokens']) for c in b['cases']]
    for x,y in zip(a['cases'],b['cases']):
     points.append({'population':manifest.parent.name+'/'+atom['id'],'test':x['test'],'tokens':x['tokens'],'delta':25000*(y['quality']-x['quality'])})
  if points:
   lo=min(points,key=lambda p:p['delta']);hi=max(points,key=lambda p:p['delta'])
   outside=edge['target']+edge['tolerance']<lo['delta'] or edge['target']-edge['tolerance']>hi['delta']
   result.append(dict(edge,available_case_pairs=len(points),minimum=lo,maximum=hi,outside_available_support=outside))
 report={'edges':result,'outside_count':sum(e['outside_available_support'] for e in result),'limits':'Each edge may have different available support. Interval inclusion is necessary only, not proof of one shared feasible distribution.'}
 (D/'support-screen.json').write_text(json.dumps(report,indent=2)+'\n')
 print('screened',len(result),'outside available support',report['outside_count'])
if __name__=='__main__':main()
