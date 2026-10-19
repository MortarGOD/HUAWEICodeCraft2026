"""Expand certificate-directed search to new true input populations."""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from tools.build_model_proxy_dataset import MODELS,build_attention_group
from tools.probe_attention_population import transform_group
DEST=ROOT/'local_distribution_fit/round012'
def main():
 torch.set_num_threads(2)
 if (DEST/'atoms.json').exists():raise SystemExit('round already exists')
 DEST.mkdir(parents=True,exist_ok=True);records=[]
 for seed in range(2026091600,2026091604):
  rows=(4,8,32,128,512);g=build_attention_group(MODELS[0],seed,'covariance',rows,'checkpoint-projection',ROOT/'local_model_tensor_cache',.05,128,4.)
  g=transform_group(g,{'tail':8,'sink':8},seed+10000);name=f'sink-input-seed{seed}';out=DEST/(name+'.pt');torch.save(g,out)
  records.append({'id':name,'path':str(out.relative_to(ROOT)),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'seed':2026090804,'input_seed':seed,'mode':'row_permute','fit_eligible':True,'profile':{'tail':8,'sink':8,'covariance_rank':128,'structured_strength':4.,'rows':rows},'model':MODELS[0].name})
 cert=json.loads((ROOT/'docs/distribution-fit/reduced-conflict-certificate.json').read_text());versions=sorted({e[k] for e in cert['constraints'] for k in ('from','to')})
 report={'round':12,'atoms':records,'required_versions':versions,'selection_evidence':'Closest fixed-certificate support is a sink population; first decoy seed mutation changes gap only from 6.705 to 6.614. Broaden true input seeds while keeping generator statistics and decoy seed fixed.','limits':'Adaptive supervised synthetic input search. Every version sees the same population; no individual-version target offsets.'}
 text=json.dumps(report,indent=2)+'\n';(DEST/'atoms.json').write_text(text);doc=ROOT/'docs/distribution-fit/round012';doc.mkdir(exist_ok=True);(doc/'atoms.json').write_text(text)
if __name__=='__main__':main()
