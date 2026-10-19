"""Expand sink/tail magnitude after an observed negative v310->v311 response."""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from tools.probe_attention_population import transform_group
DEST=ROOT/'local_distribution_fit/round004'
def main():
 torch.set_num_threads(2)
 if (DEST/'atoms.json').exists():raise SystemExit('round already exists')
 DEST.mkdir(parents=True,exist_ok=True)
 source=ROOT/'local_distribution_fit/round001/m0-base.pt';g=torch.load(source,weights_only=True);records=[]
 for i,profile in enumerate([{'tail':8,'sink':16},{'tail':32,'sink':16},{'tail':32,'sink':32},{'tail':128,'sink':32}]):
  name=f'sink{profile["sink"]}-tail{profile["tail"]}';seed=2026091100+i
  out=DEST/(name+'.pt');torch.save(transform_group(g,profile,seed),out)
  records.append({'id':name,'path':str(out.relative_to(ROOT)),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'seed':seed,'mode':'block_resample','fit_eligible':True,'source_path':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'profile':profile})
 report={'round':4,'atoms':records,'selection_evidence':'Observed round001 sink/block-resample 512-token case v310->v311 delta about -1878; target -4995. Test larger sink/tail amplitudes with unchanged judge and isolated roles.','limits':'Adaptive training support; transformed checkpoint projections are hypotheses, not captured hidden activations.'}
 (DEST/'atoms.json').write_text(json.dumps(report,indent=2)+'\n');doc=ROOT/'docs/distribution-fit/round004';doc.mkdir(exist_ok=True);(doc/'atoms.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
