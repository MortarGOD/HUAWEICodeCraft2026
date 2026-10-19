"""Fixed original integer bank; milestone estimates anchored to online v441."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'docs/distribution-fit';OUT=ROOT/'docs/gptq-integer-campaign';OUT.mkdir(exist_ok=True)
p=argparse.ArgumentParser();p.add_argument('--versions',required=True);args=p.parse_args();versions=[int(v) for v in args.versions.split(',')]
ref=ROOT/'docs/papers/hif4-public-references/test/datasets/attention_history_fit_v1/manifest.json';manifest=json.loads(ref.read_text());counts={c['label']:c['integer_count'] for c in manifest['cases']};assert sum(counts.values())==250
meta=json.loads((ROOT/'local_distribution_fit/reference-history-v1/atoms.json').read_text())['atoms']
def score(v):
 total=0.;cases=[];seconds=0.
 for a in meta:
  path=D/'reference-history-v1/cells'/f'v{v}--{a["id"]}.json';r=json.loads(path.read_text());assert r['complete'] and r['data_sha256']==a['sha256'];assert r['source_sha256']==hashlib.sha256((ROOT/f'solution/solution_v{v}.py').read_bytes()).hexdigest()
  seconds+=r['wall_seconds']
  for c,x in zip(a['reference_cases'],r['cases']):
   value=100*counts[c['label']]*x['quality'];total+=value;cases.append(dict(label=c['label'],count=counts[c['label']],quality=x['quality'],points=value))
 return total,cases,seconds
parent,pc,pt=score(441)
for v in versions:
 for a in meta:
  path=D/'reference-history-v1/cells'/f'v{v}--{a["id"]}.json'
  if path.exists():
   r=json.loads(path.read_text())
   if r.get('complete') and r['source_sha256']==hashlib.sha256((ROOT/f'solution/solution_v{v}.py').read_bytes()).hexdigest() and r['data_sha256']==a['sha256']:continue
  log=OUT/f'v{v}-{a["id"]}.log'
  with log.open('w') as f:subprocess.run([sys.executable,str(ROOT/'tools/distribution_fit/evaluate_cell.py'),'--round','reference-history-v1','--atom',a['id'],'--version',str(v)],check=True,stdout=f,stderr=subprocess.STDOUT,timeout=900,cwd=ROOT)
  print(v,a['id'],'complete',flush=True)
 total,cases,seconds=score(v);prediction=21308+total-parent
 result={'version':v,'source_sha256':hashlib.sha256((ROOT/f'solution/solution_v{v}.py').read_bytes()).hexdigest(),'parent_version':441,'parent_online_score':21308,'integer_weight_sha256':hashlib.sha256(ref.read_bytes()).hexdigest(),'weighted_case_total':250,'raw_attention_score':total,'parent_raw_attention_score':parent,'raw_delta':total-parent,'conditional_online_estimate':prediction,'thresholds_reached':[n for n in [21500,21800,22100,22400,22700,23000] if prediction>=n],'measurement_wall_seconds':seconds,'parent_measurement_wall_seconds':pt,'cases':cases,'limits':'Same frozen original integer profile. Parent-anchored local estimate, not guaranteed online score. Local timings cannot establish the 300-second online limit.'}
 (OUT/f'v{v}.json').write_text(json.dumps(result,indent=2)+'\n');print({k:z for k,z in result.items() if k!='cases'},flush=True)
