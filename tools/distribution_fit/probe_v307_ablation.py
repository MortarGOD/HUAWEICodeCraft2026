"""Separate GPTQ enabling from batched backend changes on fixed existing traces."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.distribution_fit import evaluate_cell as runner
original_load=runner.load

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--all-traces',action='store_true');args=parser.parse_args()
 out=ROOT/'docs/distribution-fit/v307-ablation';out.mkdir(exist_ok=True)
 atoms=['trace35-group9-row_permute','trace25-group40-row_permute']
 if args.all_traces:atoms=[a['id'] for a in json.loads((ROOT/'local_distribution_fit/round003/atoms.json').read_text())['atoms'] if a['mode']=='row_permute']
 modes=['native307','gptq_off307','qks_off307','both_off307','dynamic_off307'] if args.all_traces else ['parent306','native307','gptq_off307','torch307','q_only307','k_only307','qks_off307','both_off307']
 for atom in atoms:
  for mode in modes:
   def loader(version,suffix,mode=mode):
    m=original_load(version,suffix+'_'+mode)
    if mode in ['qks_off307','both_off307']:m.QKS_MODE='off'
    if mode in ['gptq_off307','both_off307']:m.ATTN_GPTQ_ENABLE=False
    if mode=='q_only307' and not (suffix=='fit_cal_q' or suffix.startswith('fit_dyn_q_')):m.ATTN_GPTQ_ENABLE=False
    if mode=='k_only307' and not (suffix=='fit_cal_k' or suffix.startswith('fit_dyn_k_')):m.ATTN_GPTQ_ENABLE=False
    if mode=='dynamic_off307':
     for role in 'qk':
      name='hif4_dynamic_quantize_'+role;original=getattr(m,name)
      def without_gptq(*args,original=original):
       state=dict(args[-1]);state.pop('gq',None)
       return original(*args[:-1],state)
      setattr(m,name,without_gptq)
    if mode=='torch307':m._gptq_quantize_batched=m._gptq_quantize_batched_torch
    return m
   runner.load=loader
   path=out/(atom+'--'+mode+'.json')
   if path.exists() and json.loads(path.read_text()).get('complete'):continue
   sys.argv=['probe','--version','306' if mode=='parent306' else '307','--round','round003','--atom',atom,'--diagnostic-output',str(path)]
   runner.main()
   r=json.loads(path.read_text());r['experimental_override']=mode;r['not_unmodified_version_measurement']=mode not in ['native307','parent306'];path.write_text(json.dumps(r,indent=2)+'\n')
 if args.all_traces:
  result=[]
  for atom in atoms:
   r={mode:json.loads((out/(atom+'--'+mode+'.json')).read_text()) for mode in modes}
   v={k:x['mean_quality'] for k,x in r.items()}
   result.append({'atom':atom,'quality':v,'fixed_state_gptq_gain':v['native307']-v['dynamic_off307'],'gptq_gain_with_qks':v['native307']-v['gptq_off307'],'gptq_gain_without_qks':v['qks_off307']-v['both_off307']})
  (out/'all-trace-factorial.json').write_text(json.dumps({'results':result,'limits':'12 preselected source groups; four configuration factorial, no online conversion or fit cells.'},indent=2)+'\n');print(result);return
 summary=[]
 for atom in atoms:
  records={mode:json.loads((out/(atom+'--'+mode+'.json')).read_text()) for mode in ['parent306','native307','gptq_off307','torch307','q_only307','k_only307','qks_off307','both_off307']}
  summary.append({'atom':atom,'quality':{k:v['mean_quality'] for k,v in records.items()},'off_vs_parent_mse_exact':all(a['player_mse']==b['player_mse'] for a,b in zip(records['parent306']['cases'],records['gptq_off307']['cases'])),'torch_vs_native_mse_exact':all(a['player_mse']==b['player_mse'] for a,b in zip(records['native307']['cases'],records['torch307']['cases']))})
 (out/'summary.json').write_text(json.dumps({'results':summary,'limits':'Two preselected existing trace groups, independent role calibration. Overrides diagnostic only; not fitting cells or global backend equivalence proof.'},indent=2)+'\n');print(summary)
if __name__=='__main__':main()
