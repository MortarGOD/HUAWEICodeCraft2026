"""Prepare only an explicitly selected complete seed family for grouped fitting."""
import argparse,json,hashlib,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/distribution-fit'
def main():
 p=argparse.ArgumentParser();p.add_argument('--family',required=True,choices=['round002','round025']);p.add_argument('--source',default='amplitude-large-first-595.json');p.add_argument('--output',required=True);args=p.parse_args()
 meta=json.loads((ROOT/'local_distribution_fit/round041/atoms.json').read_text())
 atoms=[a for a in meta['atoms'] if a['id'].startswith(args.family+'-noise005-')];assert len(atoms)==3
 assert len({a['sha256'] for a in atoms})==1 and len({a['seed'] for a in atoms})==3
 source=D/args.source;fit=json.loads(source.read_text());added=[];audits={}
 # Audit all three before writing output; never admit a partial family.
 for a in atoms:
  subprocess.run([sys.executable,str(ROOT/'tools/distribution_fit/audit_population.py'),'--round','round041','--atom',a['id'],'--integrity-only'],check=True)
  path=D/'round041'/(a['id']+'-full-audit.json');audit=json.loads(path.read_text());audits[a['id']]=hashlib.sha256(path.read_bytes()).hexdigest()
  for test,tokens in audit['case_signature']:
   label='round041/'+a['id']+f'/test{test}-tokens{tokens}'
   if label not in fit['weights']:fit['weights'][label]=0.;added.append(label)
 fit['explicit_seed_family_extension']={'family':args.family,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'audit_sha256':audits,'added_columns':added,'required_fit_option':'--seed-group-round round041','limits':'Explicit mechanism-reviewed family only. Integrity-only manifests remain excluded from broad automatic expansion. Must use grouped solver and Decimal verification; not an accepted fit.'}
 (D/args.output).write_text(json.dumps(fit,indent=2)+'\n');print('Prepared',len(added),'zero-weight measured columns')
if __name__=='__main__':main()
