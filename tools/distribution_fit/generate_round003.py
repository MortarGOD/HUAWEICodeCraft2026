"""Expand captured layer coverage without reading any candidate outputs."""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
DEST=ROOT/'local_distribution_fit/round003'
def main():
 torch.set_num_threads(2)
 if (DEST/'atoms.json').exists():raise SystemExit('round already exists')
 DEST.mkdir(parents=True,exist_ok=True);records=[]
 for label,dataset,indices in [('trace35','qwen3_5_35b_a3b_v1',(1,3,5,7,9)),('trace25','qwen2_5_14b_online_v1',(1,8,16,24,32,40,49))]:
  source=ROOT/'docs/papers/hif4-public-references/test/datasets'/dataset/'attn.pt'
  groups=torch.load(source,weights_only=True);source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
  for idx in indices:
   name=f'{label}-group{idx}';out=DEST/(name+'.pt');torch.save(groups[idx],out)
   for mode in ('correlated_noise','row_permute'):
    records.append({'id':name+'-'+mode,'path':str(out.relative_to(ROOT)),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'seed':2026091000+idx,'mode':mode,'noise':.05,'fit_eligible':True,'source_path':str(source.relative_to(ROOT)),'source_sha256':source_hash,'source_group':idx,'profile':{}})
 report={'round':3,'atoms':records,'selection_evidence':'Round002 group0 true traces changed previously uncovered directions. Expand evenly spaced source groups before assuming group0 represents all layers.','limits':'Captured source groups reused for supervised fitting; not independent validation or reconstructed hidden data.'}
 (DEST/'atoms.json').write_text(json.dumps(report,indent=2)+'\n')
 doc=ROOT/'docs/distribution-fit/round003';doc.mkdir(exist_ok=True);(doc/'atoms.json').write_text(json.dumps(report,indent=2)+'\n')
 print(len(records),'atoms')
if __name__=='__main__':main()
