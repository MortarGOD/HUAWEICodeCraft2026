"""Probe true Q/K logit scale with fixed V encoding and decoy seed."""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from tools.probe_attention_population import transform_group

def main():
 torch.set_num_threads(2)
 dest=ROOT/'local_distribution_fit/round020'
 if (dest/'atoms.json').exists():raise SystemExit('round already exists')
 source=next(a for a in json.loads((ROOT/'local_distribution_fit/round013/atoms.json').read_text())['atoms'] if a['id']=='sink-input-seed2026091600-seedplus4')
 path=ROOT/source['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
 group=torch.load(path,weights_only=True);dest.mkdir(parents=True,exist_ok=True);records=[]
 for gain in (.375,.625,.75,1.25):
  transformed=transform_group(group,{'gain':gain},2026091800)
  for phase in ('calib','test'):
   for original,current in zip(group[phase],transformed[phase]):current['v']=original['v']
  name=f'true-qk-gain{gain}';out=dest/(name+'.pt');torch.save(transformed,out)
  records.append(dict(source,id=name,path=str(out.relative_to(ROOT)),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),source_path=source['path'],source_sha256=source['sha256'],profile={'gain':gain,'v_encoding_preserved':True}))
 prices=json.loads((ROOT/'docs/distribution-fit/conflict-prices.json').read_text())
 report={'round':20,'atoms':records,'required_versions':prices['required_versions'],'selection_evidence':'Current support remains infeasible after broader layer coverage. Half-logit scale is nearest the current certificate boundary; sample adjacent previously unmeasured scales with fixed V encoding and calibration protocol.','limits':'Adaptive synthetic support search; shared inputs across all versions; no changes to score targets or acceptance.'}
 text=json.dumps(report,indent=2)+'\n';(dest/'atoms.json').write_text(text)
 doc=ROOT/'docs/distribution-fit/round020';doc.mkdir(exist_ok=True);(doc/'atoms.json').write_text(text)
if __name__=='__main__':main()
