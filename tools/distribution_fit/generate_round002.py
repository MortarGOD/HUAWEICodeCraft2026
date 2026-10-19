"""Adaptive support expansion selected from round001 direction failures."""
import copy,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from tools.probe_attention_population import transform_group
DEST=ROOT/'local_distribution_fit/round002'
def main():
 torch.set_num_threads(4)
 if (DEST/'atoms.json').exists():raise SystemExit('round already exists')
 DEST.mkdir(parents=True,exist_ok=True)
 sources=[('projected',ROOT/'local_distribution_fit/round001/m0-base.pt',None),('trace35',ROOT/'docs/papers/hif4-public-references/test/datasets/qwen3_5_35b_a3b_v1/attn.pt',0),('trace25',ROOT/'docs/papers/hif4-public-references/test/datasets/qwen2_5_14b_online_v1/attn.pt',0)]
 records=[]
 for si,(label,path,index) in enumerate(sources):
  source=torch.load(path,weights_only=True);g=source if index is None else source[index]
  for gain in (1,4):
   seed=2026090900+si*100+gain
   group=transform_group(copy.deepcopy(g),{'gain':gain},seed)
   atom=f'{label}-gain{gain}';out=DEST/(atom+'.pt');torch.save(group,out)
   for noise in (.05,.5):
    records.append({'id':atom+f'-noise{noise}','path':str(out.relative_to(ROOT)),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'seed':seed,'mode':'correlated_noise','noise':noise,'fit_eligible':True,'source_path':str(path.relative_to(ROOT)),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_group':index,'profile':{'gain':gain}})
 (DEST/'atoms.json').write_text(json.dumps({'round':2,'atoms':records,'selection_evidence':'round001 first7 populations: 36/100 edges outside convex support; v310->311 wrong direction and v439 drop too large.','limits':'Real trace groups are reused, not independent holdout. Fake peer correlation is an adjustable hypothesis; only own-role calibration state survives.'},indent=2)+'\n');print(len(records),'atoms')
if __name__=='__main__':main()
