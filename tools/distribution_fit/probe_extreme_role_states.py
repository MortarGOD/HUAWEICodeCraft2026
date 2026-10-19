"""Inspect actual independently retained role transforms on extreme fit sources."""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.distribution_fit import evaluate_cell as runner
original_load=runner.load

def main():
 out=ROOT/'docs/distribution-fit/extreme-role-states';out.mkdir(exist_ok=True)
 summary=[]
 for rnd,atom,versions in [('round003','trace35-group3-correlated_noise',[283,284]),('round021','trace35-group7-seedplus3',[322,333])]:
  for version in versions:
   retained={}
   def loader(v,suffix):
    module=original_load(v,suffix+'_state_probe')
    if suffix.startswith('fit_cal_'):
     role=suffix[-1];original=module.hif4_calibration_attention
     def calibrate(*args):
      result=original(*args)
      retained[role]=copy.deepcopy(result[role+'_state'])
      return result
     module.hif4_calibration_attention=calibrate
    return module
   runner.load=loader;dest=out/f'{rnd}-{atom}-v{version}.json'
   sys.argv=['probe','--version',str(version),'--round',rnd,'--atom',atom,'--diagnostic-output',str(dest)]
   runner.main()
   raw=json.loads(dest.read_text());prior=json.loads((ROOT/'docs/distribution-fit'/rnd/'cells'/f'v{version}--{atom}.json').read_text())
   assert raw['source_sha256']==prior['source_sha256'] and raw['data_sha256']==prior['data_sha256']
   assert all(a['player_mse']==b['player_mse'] for a,b in zip(raw['cases'],prior['cases'])), 'Diagnostic rerun drift'
   row={'round':rnd,'atom':atom,'version':version,'original_mse_equal':True,'fields':{}}
   for field in ['permutation','joint_transform']:
    q=(retained['q'] or {}).get(field);k=(retained['k'] or {}).get(field)
    item={'q_present':q is not None,'k_present':k is not None}
    if isinstance(q,torch.Tensor) and isinstance(k,torch.Tensor):
     item.update(q_shape=list(q.shape),k_shape=list(k.shape))
     if q.shape[0]%k.shape[0]==0 and q.shape[1:]==k.shape[1:]:
      aligned=k.repeat_interleave(q.shape[0]//k.shape[0],0)
      item.update(equal=bool(torch.equal(q,aligned)),different_elements=int((q!=aligned).sum()),max_abs_difference=float((q.double()-aligned.double()).abs().max()))
    row['fields'][field]=item
   summary.append(row)
   (out/'summary.json').write_text(json.dumps({'rows':summary,'limits':'Actual states and exact rerun checks. Transform inequality alone is not proof of final-error causation or online prevalence.'},indent=2)+'\n')
   print(row,flush=True)
if __name__=='__main__':main()
