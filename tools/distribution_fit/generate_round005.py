"""Persistent low-rank covariance support for naturally activating GPTQ gates."""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from tools.build_model_proxy_dataset import MODELS,build_attention_group
DEST=ROOT/'local_distribution_fit/round005'
def main():
 torch.set_num_threads(2)
 if (DEST/'atoms.json').exists():raise SystemExit('round already exists')
 DEST.mkdir(parents=True,exist_ok=True);records=[]
 for i,(rank,strength,length) in enumerate(((8,4,128),(8,16,128),(32,16,128),(8,16,32))):
  seed=2026091200+i;name=f'rank{rank}-strength{strength}-length{length}';rows=(length,)*5
  g=build_attention_group(MODELS[0],seed,'covariance',rows,'checkpoint-projection',ROOT/'local_model_tensor_cache',.05,rank,strength)
  out=DEST/(name+'.pt');torch.save(g,out)
  records.append({'id':name,'path':str(out.relative_to(ROOT)),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'seed':seed,'mode':'correlated_noise','noise':.05,'fit_eligible':True,'model':MODELS[0].name,'profile':{'covariance_rank':rank,'structured_strength':strength,'rows':rows}})
 report={'round':5,'atoms':records,'selection_evidence':'v367 isolated diagnostics found missing GPTQ states or zero accepted heads in first projected population. Test stronger persistent channel covariance and fixed lengths without forcing algorithm gates.','limits':'Synthetic checkpoint-projected training hypotheses; candidate outputs are not used in generation; no assertion that GPTQ gates activate until measured.'}
 (DEST/'atoms.json').write_text(json.dumps(report,indent=2)+'\n');doc=ROOT/'docs/distribution-fit/round005';doc.mkdir(exist_ok=True);(doc/'atoms.json').write_text(json.dumps(report,indent=2)+'\n')
 print(len(records),'atoms generated')
if __name__=='__main__':main()
