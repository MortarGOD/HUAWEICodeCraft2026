"""Fixed-input V-only mechanism probe; no score-guided input generation or fitting."""
import json,sys,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.build_fitted_dataset import quantize_nvfp4
from hif4_quant import dequantize_nvfp4,dequantize_hif4
from self_check import validate_hif4_params
@torch.inference_mode()
def main():
 torch.set_num_threads(4)
 parent=load(310,'mechanism');child=load(311,'mechanism');off=load(311,'polish_off')
 off._polish_value_mean=lambda x,values,unit,rounds=8:values
 rows=[]
 for seed in range(6):
  for tokens in [4,16,64,256]:
   for profile in ['normal','shared_mean','repeated_rows','fixed_sink']:
    gen=torch.Generator().manual_seed(2026090900+seed)
    raw=torch.randn(tokens,64,generator=gen)
    if profile=='shared_mean':raw+=2*torch.randn(1,64,generator=gen)
    if profile=='repeated_rows':raw=raw[:1].expand(tokens,-1).clone()+.03*raw
    if profile=='fixed_sink':raw[0]*=8
    q,s=quantize_nvfp4(raw);x=dequantize_nvfp4(q,s).double()
    params=[m.hif4_dynamic_quantize_v(q,s,1,64,{}) for m in [parent,child,off]]
    for p in params:assert not validate_hif4_params(p,(tokens,64),'probe')
    decoded=[dequantize_hif4(p,(tokens,64)).double() for p in params]
    assert torch.equal(decoded[0],decoded[2]),'polish-off must reproduce parent output'
    e0=x-decoded[0];e1=x-decoded[1]
    # All token selections are measured; no selection based on observed errors.
    row_delta=(e1.square()-e0.square()).mean(1)
    uniform0=e0.mean(0).square().mean();uniform1=e1.mean(0).square().mean()
    rows.append(dict(seed=seed,tokens=tokens,profile=profile,changed=int((decoded[0]!=decoded[1]).sum()),
     sse_delta=float(e1.square().mean()-e0.square().mean()),uniform_delta=float(uniform1-uniform0),
     objective_delta=float((e1.sum(0).square()+e1.square().sum(0)).sum()-(e0.sum(0).square()+e0.square().sum(0)).sum()),
     fixed_first_token_delta=float(row_delta[0]),hurt_token_fraction=float((row_delta>0).double().mean()),
     one_hot_mean_delta=float(row_delta.mean()),one_hot_max_delta=float(row_delta.max())))
 out={'source_hashes':{str(v):hashlib.sha256((ROOT/f'solution/solution_v{v}.py').read_bytes()).hexdigest() for v in [310,311]},'cases':rows,'total':len(rows),'polish_off_parent_equal':True,
 'limits':'V-only controlled diagnostic. Uniform and one-hot P are analytic weight regimes, not six-API joint Q/K measurements or reconstructed online cases. No fitting weights or online predictions.'}
 for profile in ['normal','shared_mean','repeated_rows','fixed_sink']:
  a=[r for r in rows if r['profile']==profile];print(profile,{'changed_cases':sum(r['changed']>0 for r in a),'uniform_improved':sum(r['uniform_delta']<0 for r in a),'sse_worse':sum(r['sse_delta']>0 for r in a),'fixed_first_worse':sum(r['fixed_first_token_delta']>0 for r in a),'objective_worse_after_bf16':sum(r['objective_delta']>1e-10 for r in a)})
 (ROOT/'docs/distribution-fit/v311-mechanism-probe.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
