"""Oracle-only error decomposition. Never a submission or calibration input."""
import copy,json
from pathlib import Path
import torch
from tools.test_v422 import load
from tools.check_role_decoy_calibration import role_samples
from hif4_quant import dequantize_nvfp4,dequantize_hif4,quantize_to_hif4
from evaluate_local import gqa_attention,mse,score
ROOT=Path(__file__).resolve().parents[1]
@torch.inference_mode()
def main():
 torch.set_num_threads(4)
 atoms=json.loads((ROOT/'local_distribution_fit/reference-history-v1/atoms.json').read_text())['atoms'];rows=[]
 for name in ('group15','group16'):
  atom=next(a for a in atoms if a['id']==name);assert atom['mode']=='row_permute' and not atom.get('decoy_gains') and not atom.get('role_modes')
  g=torch.load(ROOT/atom['path'],weights_only=True);qh,kh,d=(g[k] for k in ('q_num_heads','kv_num_heads','head_dim'));states={}
  for role in 'qkv':
   m=load(556,'biascal'+role);cal=role_samples(g['calib'],role,'row_permute',atom['seed']+ord(role));states[role]=m.hif4_calibration_attention(cal,qh,kh,d)[role+'_state']
  for idx,sample in enumerate(g['test']):
   raw={r:dequantize_nvfp4(*sample[r]).float() for r in 'qkv'};decoded={};params={};t=len(raw['v'])
   for role in 'qkv':
    m=load(556,'biasdyn'+role);params[role]=getattr(m,'hif4_dynamic_quantize_'+role)(*sample[role],qh if role=='q' else kh,d,copy.deepcopy(states[role]));decoded[role]=dequantize_hif4(params[role],raw[role].shape)
   def probabilities(q,k):
    q=q.reshape(t,qh,d).transpose(0,1);k=k.reshape(t,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
    return (q@k.transpose(1,2)/d**.5).softmax(-1).reshape(kh,-1,t).double()
   p=probabilities(raw['q'],raw['k']);ph=probabilities(decoded['q'],decoded['k']);v=raw['v'].reshape(t,kh,d).transpose(0,1).double();z=p@v
   h=ph.transpose(1,2)@ph/ph.shape[1];ridge=h.diagonal(dim1=-1,dim2=-2).mean(-1)*.001;hreg=h+ridge[:,None,None]*torch.eye(t)
   rhs=ph.transpose(1,2)@z/ph.shape[1]+ridge[:,None,None]*v
   adjusted=torch.linalg.solve(hreg,rhs)
   upper=torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(hreg)),upper=True).float()
   target_output=gqa_attention(*[raw[r] for r in 'qkv'],qh,kh,d)
   baseline={r:dequantize_hif4(quantize_to_hif4(raw[r]),raw[r].shape) for r in 'qkv'}
   bm=mse(target_output,gqa_attention(*[baseline[r] for r in 'qkv'],qh,kh,d))
   base_error=mse(target_output,gqa_attention(*[decoded[r] for r in 'qkv'],qh,kh,d));record={'group':name,'test':idx,'tokens':t,'current_quality':score(bm,base_error)}
   for label,target in (('oracle_metric',v),('oracle_probability_compensation',adjusted)):
    x=target.transpose(1,2).float().contiguous();units=m._params_unit_flat(params['v']).reshape(t,kh,d).permute(1,2,0).contiguous()
    q=m._gptq_quantize_batched(x,units,upper);candidate=m._values_to_params(q.permute(2,0,1).reshape(t,kh*d),params['v'])
    candidate=m._gptq_joint_metadata(target.transpose(0,1).reshape(t,kh*d).float(),candidate,hreg.float(),kh,d)
    value=dequantize_hif4(candidate,raw['v'].shape);error=mse(target_output,gqa_attention(decoded['q'],decoded['k'],value,qh,kh,d));record[label]={'quality':score(bm,error),'mse':error}
   rows.append(record);print(record,flush=True)
   (ROOT/'docs/gptq-integer-campaign/probability-bias-oracle.json').write_text(json.dumps({'diagnostic_only':True,'version':556,'complete':False,'cases':rows},indent=2)+'\n')
 (ROOT/'docs/gptq-integer-campaign/probability-bias-oracle.json').write_text(json.dumps({'diagnostic_only':True,'version':556,'complete':True,'cases':rows},indent=2)+'\n')
if __name__=='__main__':main()
