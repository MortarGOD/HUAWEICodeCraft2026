"""Raw per-head second moments, distinct from transformed GPTQ Hessians."""
import json,sys,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from hif4_quant import dequantize_nvfp4
@torch.inference_mode()
def main():
 torch.set_num_threads(4);m=json.loads((ROOT/'local_distribution_fit/round003/atoms.json').read_text());out=[]
 for name in ['trace35-group9-row_permute','trace25-group40-row_permute']:
  a=next(x for x in m['atoms'] if x['id']==name);p=ROOT/a['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==a['sha256'];g=torch.load(p,weights_only=True)
  for role in 'qk':
   heads=g['q_num_heads'] if role=='q' else g['kv_num_heads'];d=g['head_dim']
   def moment(samples):
    x=torch.cat([dequantize_nvfp4(*s[role]).double() for s in samples]).reshape(-1,heads,d).transpose(0,1)
    return x.transpose(-1,-2)@x/x.shape[1]
   cal=moment(g['calib']);test=moment(g['test']);diag=torch.diag_embed(cal.diagonal(dim1=-2,dim2=-1));norm=cal.norm(dim=(-2,-1)).clamp_min(1e-30)
   out.append({'atom':name,'role':role,'off_diagonal_energy_fraction':((cal-diag).square().sum((-2,-1))/cal.square().sum((-2,-1)).clamp_min(1e-30)).tolist(),'relative_test_moment_drift':((test-cal).norm(dim=(-2,-1))/norm).tolist()})
 path=ROOT/'docs/distribution-fit/v307-ablation/raw-moments.json';path.write_text(json.dumps({'rows':out,'limits':'Own-role raw second moments. Not transformed Hessian, fake-peer metric, or causal predictor. Two preselected groups only.'},indent=2)+'\n')
 for r in out:print(r['atom'],r['role'],'offdiag avg',sum(r['off_diagonal_energy_fraction'])/len(r['off_diagonal_energy_fraction']),'drift avg',sum(r['relative_test_moment_drift'])/len(r['relative_test_moment_drift']))
if __name__=='__main__':main()
