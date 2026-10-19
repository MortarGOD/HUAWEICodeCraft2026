"""Predeclared reversal probe of token permutation equivariance, V-only."""
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
 torch.set_num_threads(4);mods=[load(v,'perm') for v in [310,311]];rows=[]
 for seed in range(6):
  for t in [4,16,64,256]:
   for noise in [0,.03,1]:
    g=torch.Generator().manual_seed(2026090900+seed);z=torch.randn(t,64,generator=g)
    raw=z[:1].expand(t,-1).clone()+noise*z
    q,s=quantize_nvfp4(raw);x=dequantize_nvfp4(q,s).double();perm=torch.arange(t-1,-1,-1)
    outputs=[];changes=[]
    for m in mods:
     pair=[]
     for reverse in [False,True]:
      p=m.hif4_dynamic_quantize_v(q[perm] if reverse else q,s[perm] if reverse else s,1,64,{})
      assert not validate_hif4_params(p,(t,64),'perm')
      y=dequantize_hif4(p,(t,64)).double();pair.append(y[perm] if reverse else y)
     outputs.append(pair);changes.append(int((pair[0]!=pair[1]).sum()))
    base,child=outputs
    rows.append(dict(seed=seed,tokens=t,noise=noise,parent_permutation_changes=changes[0],child_permutation_changes=changes[1],
     original_first_delta=float(((x-child[0]).square()-(x-base[0]).square())[0].mean()),
     reversed_original_first_delta=float(((x-child[1]).square()-(x-base[1]).square())[0].mean()),
     original_last_delta=float(((x-child[0]).square()-(x-base[0]).square())[-1].mean())))
 summary={str(n):{'cases':len(a),'parent_non_equivariant':sum(r['parent_permutation_changes']>0 for r in a),'child_non_equivariant':sum(r['child_permutation_changes']>0 for r in a),'first_hurt':sum(r['original_first_delta']>0 for r in a),'first_hurt_after_reverse':sum(r['reversed_original_first_delta']>0 for r in a)} for n in [0,.03,1] for a in [[r for r in rows if r['noise']==n]]}
 out={'summary':summary,'cases':rows,'source_hashes':{str(v):hashlib.sha256((ROOT/f'solution/solution_v{v}.py').read_bytes()).hexdigest() for v in [310,311]},'limits':'Reversal fixed before measurements. V-only diagnostic; no QK quantization, score fitting, or claim about online token order.'}
 (ROOT/'docs/distribution-fit/v311-permutation-probe.json').write_text(json.dumps(out,indent=2)+'\n');print(summary)
if __name__=='__main__':main()
