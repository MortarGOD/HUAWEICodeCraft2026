"""Independently reconstruct a finite-support separation certificate from raw cells."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/distribution-fit'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('certificate',type=Path);args=parser.parse_args()
 report=json.loads(args.certificate.read_text());edges=report['constraints'];c=np.asarray(report['support_separation']['separating_vector'],dtype=np.longdouble)
 sources={r['version']:r for r in json.loads((D/'manifest.json').read_text())['rows']}
 atoms={}
 for p in (ROOT/'local_distribution_fit').glob('round*/atoms.json'):
  for a in json.loads(p.read_text())['atoms']:atoms[p.parent.name+'/'+a['id']]=a
 matrix=[]
 for label in report['population_ids']:
  population,case=label.rsplit('/test',1);test,tokens=map(int,case.split('-tokens'));round_name,atom_id=population.split('/',1);atom=atoms[population]
  assert atom['fit_eligible'];values={}
  for v in {e[k] for e in edges for k in ('from','to')}:
   cell=json.loads((D/round_name/f'cells/v{v}--{atom_id}.json').read_text())
   assert cell['complete'] and cell['data_sha256']==atom['sha256']
   assert cell['source_sha256']==sources[v]['source_sha256']==hashlib.sha256((ROOT/f'solution/solution_v{v}.py').read_bytes()).hexdigest()
   item=next(x for x in cell['cases'] if x['test']==test);assert item['tokens']==tokens
   values[v]=np.longdouble(item['quality'])*25000
  matrix.append([values[e['to']]-values[e['from']] for e in edges])
 A=np.asarray(matrix,dtype=np.longdouble).T
 lo=np.array([e['target']-e['tolerance'] for e in edges],dtype=np.longdouble);hi=np.array([e['target']+e['tolerance'] for e in edges],dtype=np.longdouble)
 for i,e in enumerate(edges):
  if e['target']>1:lo[i]=max(lo[i],1e-9)
  if e['target']< -1:hi[i]=min(hi[i],-1e-9)
 lower=np.sum(np.where(c>=0,c*lo,c*hi));upper=np.max(c@A);gap=lower-upper
 guard=1e-10*(1+np.sum(np.abs(c)*np.maximum(np.abs(lo),np.abs(hi)))+np.max(np.abs(c)@np.abs(A)))
 assert gap>guard,(gap,guard)
 print(json.dumps({'verified':True,'columns':len(matrix),'constraints':len(edges),'gap':float(gap),'numerical_guard':float(guard),'scope':'Only listed complete columns, reconstructed from unchanged source/data-identified measurement cells.'}))
if __name__=='__main__':main()
