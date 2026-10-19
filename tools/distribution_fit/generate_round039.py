"""Predefined reciprocal channel anisotropy for actual 412/417 mechanism tests."""
import copy,json,hashlib,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.build_fitted_dataset import quantize_nvfp4
from hif4_quant import dequantize_nvfp4
from evaluate_local import gqa_attention,mse
@torch.inference_mode()
def main():
 torch.set_num_threads(4);dest=ROOT/'local_distribution_fit/round039';assert not dest.exists();dest.mkdir();records=[]
 manifest=json.loads((ROOT/'local_distribution_fit/round003/atoms.json').read_text())
 for source_id in ['trace35-group9-row_permute','trace25-group40-row_permute']:
  a=next(x for x in manifest['atoms'] if x['id']==source_id);p=ROOT/a['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==a['sha256'];g=torch.load(p,weights_only=True);qh,kh,d=[g[k] for k in ['q_num_heads','kv_num_heads','head_dim']]
  for strength in [1,4,16]:
   bank=copy.deepcopy(g);checks=[];scale=torch.ones(d,dtype=torch.float64);scale[:d//4]=strength;scale[d//4:d//2]=1/strength
   for phase in ['calib','test']:
    for old,new in zip(g[phase],bank[phase]):
     q,k,v=[dequantize_nvfp4(*old[r]).float() for r in 'qkv']
     qr=(q.double().reshape(-1,qh,d)*scale).reshape_as(q).float();kr=(k.double().reshape(-1,kh,d)/scale).reshape_as(k).float()
     target=gqa_attention(q,k,v,qh,kh,d);continuous=gqa_attention(qr,kr,v,qh,kh,d)
     if strength!=1:new['q']=quantize_nvfp4(qr);new['k']=quantize_nvfp4(kr)
     encoded=gqa_attention(dequantize_nvfp4(*new['q']).float(),dequantize_nvfp4(*new['k']).float(),v,qh,kh,d)
     checks.append({'phase':phase,'continuous_mse':mse(target,continuous),'encoding_mse':mse(target,encoded)})
   name=source_id.split('-row_')[0]+f'-anisotropy{strength}';out=dest/(name+'.pt');torch.save(bank,out)
   records.append(dict(a,id=name,path=str(out.relative_to(ROOT)),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),fit_eligible=False,mechanism_strength=strength,reference_checks=checks))
 report={'round':39,'atoms':records,'limits':'Reciprocal QK scaling preserves pre-encoding product; NVFP4 re-encoding can change inputs, explicitly measured. Fixed V, fixed source, no outcome-guided strengths. Mechanism only, no fitting promotion.'}
 doc=ROOT/'docs/distribution-fit/round039';doc.mkdir();text=json.dumps(report,indent=2)+'\n';(dest/'atoms.json').write_text(text);(doc/'atoms.json').write_text(text);print('six fixed conditions generated')
if __name__=='__main__':main()
