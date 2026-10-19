"""Evaluate the materialized reference tensors, without assuming upstream identity."""
import json,hashlib,subprocess,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/distribution-fit';REF=ROOT/'docs/papers/hif4-public-references/test/datasets/attention_history_fit_v1'
rnd='reference-history-v1';bank=ROOT/'local_distribution_fit'/rnd;doc=D/rnd;bank.mkdir(exist_ok=True);doc.mkdir(exist_ok=True)
manifest=json.loads((REF/'manifest.json').read_text());artifact=REF/'attn.pt';assert hashlib.sha256(artifact.read_bytes()).hexdigest()==manifest['artifact']['sha256']
groups=torch.load(artifact,weights_only=True);atoms=[]
for i,g in enumerate(groups):
 path=bank/f'group{i:02d}.pt'
 if not path.exists():torch.save(g,path)
 saved=torch.load(path,weights_only=True)
 for section in ['calib','test']:
  assert len(saved[section])==len(g[section])
  for a,b in zip(saved[section],g[section]):
   for role in 'qkv':assert all(torch.equal(x,y) for x,y in zip(a[role],b[role]))
 a=dict(g['fit_protocol']);a.update(id=f'group{i:02d}',path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),fit_eligible=False,reference_cases=g['fit_cases']);atoms.append(a)
(bank/'atoms.json').write_text(json.dumps({'atoms':atoms},indent=2)+'\n')
for a in atoms:
 for v in [435,441,442,443,444]:
  dest=doc/'cells'/f'v{v}--{a["id"]}.json'
  if dest.exists():
   r=json.loads(dest.read_text())
   if r.get('complete') and r['data_sha256']==a['sha256'] and r['source_sha256']==hashlib.sha256((ROOT/f'solution/solution_v{v}.py').read_bytes()).hexdigest():continue
  logs=doc/'logs';logs.mkdir(exist_ok=True)
  with (logs/f'v{v}--{a["id"]}.log').open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools/distribution_fit/evaluate_cell.py'),'--round',rnd,'--atom',a['id'],'--version',str(v)],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=900,cwd=ROOT)
  print(a['id'],v,'complete',flush=True)
profiles={'public_integer':{c['label']:c['integer_count'] for c in manifest['cases']},'reference_default':json.loads((REF/'local-online-weights.json').read_text())['counts']}
online={435:21304,441:21308,442:21305,443:21304,444:21304};result={}
for name,counts in profiles.items():
 assert sum(counts.values())==250
 scores={v:0. for v in online}
 for a in atoms:
  for v in online:
   r=json.loads((doc/'cells'/f'v{v}--{a["id"]}.json').read_text())
   for c,measured in zip(a['reference_cases'],r['cases']):scores[v]+=100*counts[c['label']]*measured['quality']
 result[name]={'attention_scores':scores,'rows':[{'version':v,'online_total':online[v],'online_delta':online[v]-online[435],'local_delta':scores[v]-scores[435],'direct_total_linear4450':4450+scores[v],'parent_anchored_total':online[435]+scores[v]-scores[435]} for v in [441,442,443,444]]}
(doc/'new-receipt-evaluation.json').write_text(json.dumps({'artifact_sha256':manifest['artifact']['sha256'],'profiles':result,'limits':'Exact materialized tensors evaluated for five versions. Historical original-bank matches are separate; new online receipts not used to refit weights. Anchored totals are diagnostics, not independent forecasts.'},indent=2)+'\n');print(result)
