"""Common orthogonal QK rotation; fixed V, same continuous Attention target."""
import json,sys,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from hif4_quant import dequantize_nvfp4,dequantize_hif4,quantize_to_hif4
from evaluate_local import gqa_attention,mse
@torch.inference_mode()
def main():
 torch.set_num_threads(4);atoms=json.loads((ROOT/'local_distribution_fit/round003/atoms.json').read_text())['atoms'];seen=set();rows=[]
 for a in atoms:
  if a['path'] in seen:continue
  seen.add(a['path']);p=ROOT/a['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==a['sha256'];g=torch.load(p,weights_only=True)
  qh,kh,d=[g[k] for k in ['q_num_heads','kv_num_heads','head_dim']];assert d&(d-1)==0
  h=torch.ones(1,1,dtype=torch.float64)
  while h.shape[0]<d:h=torch.cat([torch.cat([h,h],1),torch.cat([h,-h],1)],0)
  h/=d**.5;assert torch.allclose(h@h.T,torch.eye(d,dtype=torch.float64),atol=1e-12)
  for i,s in enumerate(g['test']):
   q,k,v=[dequantize_nvfp4(*s[r]).float() for r in 'qkv']
   qr=(q.double().reshape(-1,qh,d)@h).reshape_as(q).float();kr=(k.double().reshape(-1,kh,d)@h).reshape_as(k).float()
   target=gqa_attention(q,k,v,qh,kh,d);continuous=gqa_attention(qr,kr,v,qh,kh,d)
   quant=lambda x:dequantize_hif4(quantize_to_hif4(x),x.shape)
   vhat=quant(v)
   raw=mse(target,gqa_attention(quant(q),quant(k),vhat,qh,kh,d))
   rotated=mse(target,gqa_attention(quant(qr),quant(kr),vhat,qh,kh,d))
   peak=lambda x:float((x.reshape(-1,d).abs().amax(-1)/x.reshape(-1,d).square().mean(-1).sqrt().clamp_min(1e-20)).mean())
   rows.append({'path':a['path'],'test':i,'tokens':len(q),'head_dim':d,'raw_mse':raw,'rotated_mse':rotated,'continuous_mse':mse(target,continuous),'q_peak_rms_raw':peak(q),'q_peak_rms_rotated':peak(qr),'k_peak_rms_raw':peak(k),'k_peak_rms_rotated':peak(kr)})
 out={'cases':rows,'improved':sum(r['rotated_mse']<r['raw_mse'] for r in rows),'worse':sum(r['rotated_mse']>r['raw_mse'] for r in rows),'limits':'Plain common Hadamard plus standard HiF4 quantizer, not adaptive V412 or migrated V417 metric. Continuous calculation differs by floating rounding. Existing selected traces, no online prevalence or score prediction.'}
 (ROOT/'docs/distribution-fit/common-rotation-probe.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='cases'});print('max continuous MSE',max(r['continuous_mse'] for r in rows))
if __name__=='__main__':main()
