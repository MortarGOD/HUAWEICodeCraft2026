"""Version-independent, seeded population atoms from cached checkpoint projections."""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from tools.build_model_proxy_dataset import MODELS,build_attention_group
from tools.probe_attention_population import transform_group
DEST=ROOT/'local_distribution_fit/round001'
def main():
 torch.set_num_threads(4)
 if (DEST/'atoms.json').exists():raise SystemExit('bank exists; choose a new round for changed parameters')
 DEST.mkdir(parents=True,exist_ok=True)
 profiles=[('base',{}),('diffuse',{'gain':.25}),('sharp',{'gain':4}),('vtail',{'tail':32,'roles':'v'}),('sink',{'tail':8,'sink':8}),('vmean',{'vmean':4})]
 records=[]
 for mi,model in enumerate(MODELS):
  for pi,(label,profile) in enumerate(profiles):
   seed=2026090800+mi*100+pi;rows=(4,8,32,128,512)
   g=build_attention_group(model,seed,'covariance',rows,'checkpoint-projection',ROOT/'local_model_tensor_cache',.05,128,4.)
   g=transform_group(g,profile,seed+10000)
   atom=f'm{mi}-{label}';path=DEST/(atom+'.pt');torch.save(g,path)
   record={'id':atom,'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'seed':seed,'profile':profile,'model':model.name,'rows':rows,'mode':('paired','row_permute','block_resample')[pi%3]}
   for mode in ('paired','row_permute','block_resample'):
    records.append(dict(record,id=atom+'-'+mode,mode=mode,fit_eligible=mode!='paired'))
   print(atom,flush=True)
 (DEST/'atoms.json').write_text(json.dumps({'round':1,'atoms':records,'limits':'12 new populations crossed with three calibration protocols, 36 atoms. Paired atoms diagnostic only; not reconstructed hidden data.'},indent=2)+'\n')
if __name__=='__main__':main()
