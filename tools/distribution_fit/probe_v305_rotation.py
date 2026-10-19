"""Decompose rotation orientation and guard size on high-contribution inputs."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.distribution_fit import evaluate_cell as runner
original_load=runner.load

def main():
 out=ROOT/'docs/distribution-fit/v305-rotation';out.mkdir(exist_ok=True);summary=[]
 for rnd,atom in [('round025','qk-gain055'),('round002','trace35-gain1-noise0.05')]:
  records={}
  for mode in ['parent303','native305','guard128','premix','forceoff303','forceon303']:
   def loader(version,suffix,mode=mode):
    m=original_load(version,suffix+'_'+mode)
    if mode in ['forceoff303','forceon303']:
     for role in 'qk':
      name='hif4_dynamic_quantize_'+role;original=getattr(m,name)
      def forced(*args,original=original):
       state=dict(args[-1]);state['rot']=int(mode=='forceon303');state['rseed']=0
       return original(*args[:-1],state)
      setattr(m,name,forced)
    if mode=='guard128':m.ATTN_ROTATION_GUARD_ROWS=128
    if mode=='premix':m._make_R=original_load(303,suffix+'_rotation')._make_R
    return m
   runner.load=loader;dest=out/(rnd+'-'+atom+'-'+mode+'.json')
   if dest.exists() and json.loads(dest.read_text()).get('complete'):
    records[mode]=json.loads(dest.read_text());continue
   sys.argv=['probe','--version','303' if mode in ['parent303','forceoff303','forceon303'] else '305','--round',rnd,'--atom',atom,'--diagnostic-output',str(dest)]
   runner.main();r=json.loads(dest.read_text());r['experimental_override']=mode;r['limits']='Selected to diagnose existing fit contributions, not independent validation.';dest.write_text(json.dumps(r,indent=2)+'\n');records[mode]=r
  summary.append({'round':rnd,'atom':atom,'quality':{k:r['mean_quality'] for k,r in records.items()},'guard128_vs_native_mse_equal':all(x['player_mse']==y['player_mse'] for x,y in zip(records['native305']['cases'],records['guard128']['cases'])),'premix_vs_parent_mse_equal':all(x['player_mse']==y['player_mse'] for x,y in zip(records['parent303']['cases'],records['premix']['cases']))})
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
if __name__=='__main__':main()
