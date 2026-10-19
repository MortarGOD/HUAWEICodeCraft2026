"""Fixed-test, predeclared peer-noise sweep; includes zero-noise re-encoding control."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.distribution_fit import evaluate_cell as runner

def main():
 bank=ROOT/'local_distribution_fit/round041';doc=ROOT/'docs/distribution-fit/round041';assert not bank.exists();bank.mkdir();doc.mkdir();atoms=[]
 for rnd,name in [('round002','trace35-gain1-noise0.05'),('round025','qk-gain055')]:
  a=next(x for x in json.loads((ROOT/'local_distribution_fit'/rnd/'atoms.json').read_text())['atoms'] if x['id']==name)
  assert hashlib.sha256((ROOT/a['path']).read_bytes()).hexdigest()==a['sha256']
  for label,mode,noise in [('noise005','correlated_noise',.005),('noise05','correlated_noise',.05),('noise20','correlated_noise',.2)]:
   for offset in [101,202,303]:
    atoms.append(dict(a,id=rnd+'-'+label+'-seed'+str(offset),seed=a['seed']+offset,mode=mode,noise=noise,fit_eligible=False))
 report={'round':41,'atoms':atoms,'limits':'Same immutable source bytes per family, all test tensors fixed. Three predeclared seed offsets101/202/303; all strengths retained. Prior seed used for hypothesis discovery only. Original round025 row_permute not part of continuous noise axis. Not an online distribution claim.'}
 text=json.dumps(report,indent=2)+'\n';(bank/'atoms.json').write_text(text);(doc/'atoms.json').write_text(text)
 rows=[]
 for a in atoms:
  for v in [303,305]:
   dest=doc/(a['id']+f'-v{v}.json');sys.argv=['probe','--version',str(v),'--round','round041','--atom',a['id'],'--diagnostic-output',str(dest)];runner.main()
   r=json.loads(dest.read_text());flags=r['calibration_diagnostics'];rows.append({'atom':a['id'],'version':v,'quality':r['mean_quality'],'q_rot':flags['q']['state_flags']['q'].get('rot'),'k_rot':flags['k']['state_flags']['k'].get('rot')})
 (doc/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows)
if __name__=='__main__':main()
