"""Measure every positive-weight atom for explicitly requested prediction versions."""
import json,hashlib,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/distribution-fit'
fit=D/'amplitude-20pct-vmean-growth.json';r=json.loads(fit.read_text())
atoms=sorted({tuple(k.split('/')[:2]) for k,w in r['weights'].items() if w>0})
out=D/'prediction-441-444';out.mkdir(exist_ok=True)
for rnd,atom in atoms:
 for v in [441,442,443,444]:
  path=D/rnd/'cells'/f'v{v}--{atom}.json';source=ROOT/f'solution/solution_v{v}.py'
  if path.exists():
   prior=json.loads(path.read_text())
   if prior.get('complete') and prior['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest():continue
  with (out/f'{rnd}-{atom}-v{v}.log').open('w') as log:
   subprocess.run([sys.executable,str(ROOT/'tools/distribution_fit/evaluate_cell.py'),'--round',rnd,'--atom',atom,'--version',str(v)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900,cwd=ROOT)
  print(rnd,atom,v,'complete',flush=True)
(out/'measurement-complete.json').write_text(json.dumps({'fit_sha256':hashlib.sha256(fit.read_bytes()).hexdigest(),'atoms':len(atoms),'versions':[441,442,443,444]})+'\n')
