"""Isolated six-API Attention measurement for a fixed source/population pair."""
import argparse,copy,hashlib,json,math,sys,time,traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.check_role_decoy_calibration import role_samples
from tools.verify_attention_campaign import equal
from hif4_quant import dequantize_nvfp4,dequantize_hif4,quantize_to_hif4
from evaluate_local import gqa_attention,mse,score
from self_check import validate_attention_calibration_result,validate_hif4_params
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--version',type=int,required=True);p.add_argument('--atom',required=True);p.add_argument('--round',default='round001');p.add_argument('--diagnostic-output',type=Path);a=p.parse_args()
 torch.set_num_threads(4);torch.manual_seed(20260908)
 bank=ROOT/'local_distribution_fit'/a.round;atom=next(x for x in json.loads((bank/'atoms.json').read_text())['atoms'] if x['id']==a.atom)
 assert a.diagnostic_output is not None, 'Oracle results must use separate diagnostic output'
 source=ROOT/f'solution/solution_v{a.version}.py';path=ROOT/atom['path']
 dest=ROOT/'docs/distribution-fit'/a.round/'cells';dest.mkdir(parents=True,exist_ok=True);output=dest/f'v{a.version}--{a.atom}.json'
 if a.diagnostic_output is not None:
  output=a.diagnostic_output;output.parent.mkdir(parents=True,exist_ok=True)
 record={'version':a.version,'atom':a.atom,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'data_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'complete':False,'cases':[],'format_checks':0}
 assert record['data_sha256']==atom['sha256']
 started=time.perf_counter()
 try:
  g=torch.load(path,weights_only=True);frozen=copy.deepcopy(g);qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')];states={}
  for role in 'qkv':
   m=load(a.version,'fit_cal_'+role)
   role_mode=atom.get('role_modes',{}).get(role,atom['mode'])
   samples=copy.deepcopy(g['calib']) if role_mode in ('paired','correlated_noise') else role_samples(g['calib'],role,role_mode,atom['seed']+ord(role))
   if role_mode=='correlated_noise':
    from tools.build_fitted_dataset import quantize_nvfp4
    generator=torch.Generator().manual_seed(atom['seed']+ord(role))
    samples=copy.deepcopy(g['calib'])
    for sample in samples:
     for peer in 'qkv':
      if peer==role:continue
      raw=dequantize_nvfp4(*sample[peer]).float()
      scale=raw.square().mean(0,keepdim=True).sqrt().clamp_min(1e-8)
      sample[peer]=quantize_nvfp4(raw+atom['noise']*scale*torch.randn(raw.shape,generator=generator))
   gains=atom.get('decoy_gains',{}).get(role,{})
   if gains:
    from tools.build_fitted_dataset import quantize_nvfp4
    for peer,gain in gains.items():
     assert peer in 'qkv' and peer!=role,'Cannot alter true calibration role'
     for sample in samples:sample[peer]=quantize_nvfp4(dequantize_nvfp4(*sample[peer]).float()*gain)
   calibrated=m.hif4_calibration_attention(samples,qh,kh,d)
   assert not validate_attention_calibration_result(calibrated,'fit')
   states[role+'_state']=copy.deepcopy(calibrated[role+'_state'])
   if a.diagnostic_output is not None:
    flags={r:{k:v for k,v in (calibrated[r+'_state'] or {}).items() if isinstance(v,(str,int,float,bool,type(None)))} for r in 'qkv'}
    debug={name:{k:v for k,v in obj.items() if isinstance(v,(str,int,float,bool,type(None)))} for name,obj in vars(m).items() if name.endswith('DEBUG') and isinstance(obj,dict)}
    for values in list(flags.values())+list(debug.values()):
     for key,value in values.items():
      if isinstance(value,float) and not math.isfinite(value):values[key]=repr(value)
    record.setdefault('calibration_diagnostics',{})[role]={'state_flags':flags,'debug':debug}
  record['calibration_seconds']=time.perf_counter()-started
  for i,s in enumerate(g['test']):
   raw={r:dequantize_nvfp4(*s[r]).float() for r in 'qkv'};deployed={};baseline={}
   for role in 'qkv':
    m=load(a.version,f'fit_dyn_{role}_{i}');st=copy.deepcopy(states[role+'_state']);saved=copy.deepcopy(st)
    qp=getattr(m,'hif4_dynamic_quantize_'+role)(*s[role],qh if role=='q' else kh,d,st)
    assert not validate_hif4_params(qp,raw[role].shape,role);record['format_checks']+=1
    assert equal(st,saved)
    deployed[role]=dequantize_hif4(qp,raw[role].shape)
    baseline[role]=dequantize_hif4(quantize_to_hif4(raw[role]),raw[role].shape)
   target=gqa_attention(*[raw[r] for r in 'qkv'],qh,kh,d)
   bm=mse(target,gqa_attention(*[baseline[r] for r in 'qkv'],qh,kh,d));pm=mse(target,gqa_attention(*[deployed[r] for r in 'qkv'],qh,kh,d))
   value=score(bm,pm);assert torch.isfinite(torch.tensor(value))
   record['cases'].append({'test':i,'tokens':len(raw['v']),'baseline_mse':bm,'player_mse':pm,'quality':value})
   oracle_qk = mse(target,gqa_attention(raw['q'],raw['k'],deployed['v'],qh,kh,d))
   oracle_v = mse(target,gqa_attention(deployed['q'],deployed['k'],raw['v'],qh,kh,d))
   record['cases'][-1].update(perfect_qk_quality=score(bm,oracle_qk),perfect_v_quality=score(bm,oracle_v),diagnostic_only=True)

  assert equal(g,frozen)
  record.update(complete=True,mean_quality=sum(x['quality'] for x in record['cases'])/len(record['cases']))
 except Exception:
  record['error']=traceback.format_exc()
 record['wall_seconds']=time.perf_counter()-started
 temp=output.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(output)
 print(json.dumps({k:record.get(k) for k in ('version','atom','complete','mean_quality','wall_seconds','error')}),flush=True)
 if not record['complete']:raise SystemExit(1)
if __name__=='__main__':main()
