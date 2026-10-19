"""Controlled Q/K canonicalization roles on unchanged source traces."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.distribution_fit import evaluate_cell as runner
original_load=runner.load

def main():
 out=ROOT/'docs/distribution-fit/v417-roles';out.mkdir(exist_ok=True)
 atoms=['trace35-group9-row_permute','trace25-group40-row_permute']
 for atom in atoms:
  for mode in ['parent412','native417','q_only417','k_only417','off417']:
   def loader(version,suffix,mode=mode):
    m=original_load(version,suffix+'_'+mode)
    if mode in ['q_only417','k_only417','off417']:
     original=m._cc_state
     def convert(state,role,kh,d):
      if mode=='off417' or role!=mode[0]:return state
      return original(state,role,kh,d)
     m._cc_state=convert
    return m
   runner.load=loader;path=out/(atom+'--'+mode+'.json')
   sys.argv=['probe','--version','412' if mode=='parent412' else '417','--round','round003','--atom',atom,'--diagnostic-output',str(path)]
   runner.main();r=json.loads(path.read_text());r['experimental_override']=mode;r['limits']='Single-role canonicalization can break continuous coordinate compatibility; diagnostic only, never a candidate or fitting label.';path.write_text(json.dumps(r,indent=2)+'\n')
 rows=[]
 for a in atoms:
  r={m:json.loads((out/(a+'--'+m+'.json')).read_text()) for m in ['parent412','native417','q_only417','k_only417','off417']}
  assert all(x['player_mse']==y['player_mse'] for x,y in zip(r['parent412']['cases'],r['off417']['cases']))
  rows.append({'atom':a,'off_parent_mse_exact':True,'quality':{m:x['mean_quality'] for m,x in r.items()}})
 (out/'summary.json').write_text(json.dumps({'rows':rows,'limits':'Role ablations are not isolated quantization effects because one-sided transforms may alter unquantized logits. No online magnitude estimate.'},indent=2)+'\n');print(rows)
if __name__=='__main__':main()
