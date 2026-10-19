"""Resumable isolated measurements; never impute failed/missing version cells."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/distribution-fit'
def main():
 p=argparse.ArgumentParser();p.add_argument('--round',default='round001');p.add_argument('--versions');p.add_argument('--atoms',help='Comma-separated exact atom ids');p.add_argument('--max-cells',type=int);p.add_argument('--timeout',type=int,default=900);a=p.parse_args()
 manifest=json.loads((D/'manifest.json').read_text());versions=[x for x in manifest['rows'] if x['included'] and (not a.versions or str(x['version']) in a.versions.split(','))]
 atoms=json.loads((ROOT/'local_distribution_fit'/a.round/'atoms.json').read_text())['atoms'];
 if a.atoms:
  wanted=set(a.atoms.split(','));assert wanted<={x['id'] for x in atoms},'unknown atom id'
  atoms=[x for x in atoms if x['id'] in wanted]
 atoms.sort(key=lambda x:not x['fit_eligible'])
 logs=D/a.round/'logs';logs.mkdir(parents=True,exist_ok=True);count=0
 for atom in atoms:
  for row in versions:
   v=row['version'];dest=D/a.round/f'cells/v{v}--{atom["id"]}.json'
   if dest.exists():
    prior=json.loads(dest.read_text())
    if prior.get('source_sha256')==row['source_sha256'] and prior.get('data_sha256')==atom['sha256']:
     continue # failures retained for explicit diagnosis, never auto-relabelled success
   assert hashlib.sha256((ROOT/f'solution/solution_v{v}.py').read_bytes()).hexdigest()==row['source_sha256']
   started=time.time();status={'active_version':v,'active_atom':atom['id'],'started_unix':started,'completed_this_run':count,'total_expected_cells':len(versions)*len(atoms)}
   (D/a.round/('progress-'+hashlib.sha256((a.atoms+'|'+str(a.versions)).encode()).hexdigest()[:12]+'.json' if a.atoms else 'progress.json')).write_text(json.dumps(status,indent=2)+'\n')
   with (logs/f'v{v}--{atom["id"]}.log').open('w') as log:
    try:
     result=subprocess.run([sys.executable,str(ROOT/'tools/distribution_fit/evaluate_cell.py'),'--version',str(v),'--atom',atom['id'],'--round',a.round],stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout,cwd=ROOT)
     state='ok' if result.returncode==0 else 'error'
    except subprocess.TimeoutExpired:
     state='timeout';dest.parent.mkdir(exist_ok=True)
     dest.write_text(json.dumps({'version':v,'atom':atom['id'],'complete':False,'error':'local cell timeout','source_sha256':row['source_sha256'],'data_sha256':atom['sha256']},indent=2)+'\n')
   count+=1;print(v,atom['id'],state,round(time.time()-started,2),flush=True)
   if a.max_cells is not None and count>=a.max_cells:return
  if a.round=='round001':subprocess.run([sys.executable,str(ROOT/'tools/distribution_fit/fit.py')],cwd=ROOT,check=True)
 (D/a.round/('progress-'+hashlib.sha256((a.atoms+'|'+str(a.versions)).encode()).hexdigest()[:12]+'.json' if a.atoms else 'progress.json')).write_text(json.dumps({'terminal':True,'reason':'all scheduled cells attempted; consult fit and failures, not goal completion'})+'\n')
if __name__=='__main__':main()
