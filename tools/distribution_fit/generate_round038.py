"""Mechanism-predeclared repeated V / fixed key sink, with equivalent KV reversal."""
import copy,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.build_fitted_dataset import quantize_nvfp4
from hif4_quant import dequantize_nvfp4
from evaluate_local import gqa_attention
@torch.inference_mode()
def main():
 torch.set_num_threads(4);dest=ROOT/'local_distribution_fit/round038';assert not dest.exists();dest.mkdir();atoms=[];equiv=[]
 for noise in [0,.03]:
  g={'q_num_heads':1,'kv_num_heads':1,'head_dim':64,'calib':[],'test':[]}
  for phase in ['calib','test']:
   for i in range(5):
    rng=torch.Generator().manual_seed(2026091100+i+(100 if phase=='test' else 0));t=16
    direction=torch.randn(1,64,generator=rng);direction/=direction.square().mean().sqrt()
    q=1.5*direction.expand(t,-1)+.03*torch.randn(t,64,generator=rng)
    k=.1*torch.randn(t,64,generator=rng);k[0]=1.5*direction[0]
    z=torch.randn(t,64,generator=rng);v=z[:1].expand(t,-1).clone()+noise*z
    g[phase].append({r:quantize_nvfp4(x) for r,x in zip('qkv',[q,k,v])})
  for rev in [False,True]:
   current=copy.deepcopy(g)
   if rev:
    for phase in ['calib','test']:
     for a,b in zip(g[phase],current[phase]):
      for r in 'kv':b[r]=[x.flip(0).contiguous() for x in a[r]]
      raws=lambda s:[dequantize_nvfp4(*s[r]).float() for r in 'qkv']
      y=gqa_attention(*raws(a),1,1,64);yp=gqa_attention(*raws(b),1,1,64)
      assert torch.allclose(y,yp,rtol=1e-5,atol=1e-6)
      equiv.append(float((y-yp).abs().max()))
   name=f'repeat{int(noise*100):02d}-'+('reverse' if rev else 'first');p=dest/(name+'.pt');torch.save(current,p)
   for mode in ['paired','row_permute']:
    atoms.append({'id':name+'-'+mode,'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'mode':mode,'fit_eligible':False,'seed':2026091100,'noise':noise,'profile':{'tokens':16,'fixed_key_sink':True,'repeated_v_noise':noise,'reverse_kv':rev}})
 doc=ROOT/'docs/distribution-fit/round038';doc.mkdir()
 out={'round':38,'atoms':atoms,'reference_reversal_max_difference':max(equiv),'mechanism':'V311 argmin token-position sensitivity on repeated/near-repeated V; fixed first key preference, no selection using quantizer errors. Calibration/test RNG separated.','limits':'Controlled synthetic probe, not online distribution evidence; measure 310/311 first, no fit promotion without mechanism result.'}
 text=json.dumps(out,indent=2)+'\n';(dest/'atoms.json').write_text(text);(doc/'atoms.json').write_text(text)
 print('8 atoms; reference reversal max difference',max(equiv))
if __name__=='__main__':main()
