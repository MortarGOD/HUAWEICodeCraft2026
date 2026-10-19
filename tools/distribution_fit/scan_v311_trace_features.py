"""Inspect fixed trace features without selecting or changing data using errors."""
import json,hashlib,sys,math
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from hif4_quant import dequantize_nvfp4
@torch.inference_mode()
def main():
 torch.set_num_threads(4);d=ROOT/'docs/distribution-fit';m=json.loads((ROOT/'local_distribution_fit/round003/atoms.json').read_text());seen=set();rows=[];skipped=[]
 for a in m['atoms']:
  if a['path'] in seen:continue
  seen.add(a['path']);p=ROOT/a['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==a['sha256'];g=torch.load(p,weights_only=True)
  qh,kh,hd=[g[k] for k in ['q_num_heads','kv_num_heads','head_dim']]
  for i,s in enumerate(g['test']):
   q,k,v=[dequantize_nvfp4(*s[r]).float() for r in 'qkv'];t=len(k)
   if len(q)*t*qh>16000000:skipped.append({'path':a['path'],'test':i,'reason':'attention matrix exceeds predeclared 16M elements'});continue
   qq=q.reshape(len(q),qh,hd).transpose(0,1);kk=k.reshape(t,kh,hd).transpose(0,1).repeat_interleave(qh//kh,0)
   prob=torch.softmax((qq@kk.transpose(-1,-2))/math.sqrt(hd),-1)
   centered=(v-v.mean(0)).square().mean();energy=v.square().mean().clamp_min(1e-30)
   r={'path':a['path'],'source_group':a['source_group'],'test':i,'tokens':t,'q_heads':qh,'kv_heads':kh,
      'v_centered_energy_fraction':float(centered/energy),'v_equal_first_fraction':float((v==v[:1]).float().mean()),
      'first_key_mass':float(prob[...,0].mean()),'first_eight_key_mass':float(prob[...,:min(8,t)].sum(-1).mean()),
      'mean_max_probability':float(prob.max(-1).values.mean()),'measurements':[]}
   for atom in m['atoms']:
    if atom['path']!=a['path']:continue
    paths=[d/'round003/cells'/f"v{version}--{atom['id']}.json" for version in [310,311]]
    if not all(p.exists() for p in paths):continue
    cells=[json.loads(p.read_text()) for p in paths]
    if not all(c['complete'] and c['data_sha256']==a['sha256'] for c in cells):continue
    cases=[next(z for z in c['cases'] if z['test']==i) for c in cells]
    r['measurements'].append({'mode':atom['mode'],'mse_delta':cases[1]['player_mse']-cases[0]['player_mse'],'quality_delta':cases[1]['quality']-cases[0]['quality']})
   rows.append(r)
 out={'cases':rows,'skipped':skipped,'limits':'Selected existing round003 source groups only; not a random online sample. Features measured before joining historical results; no data modified. Matrix cap fixed at16M elements.'}
 (d/'v311-natural-trace-features.json').write_text(json.dumps(out,indent=2)+'\n')
 print('cases',len(rows),'skipped',len(skipped))
 print('low variation',sum(r['v_centered_energy_fraction']<.05 for r in rows),'first mass>.5',sum(r['first_key_mass']>.5 for r in rows),'both',sum(r['v_centered_energy_fraction']<.05 and r['first_key_mass']>.5 for r in rows))
 for r in sorted(rows,key=lambda x:x['v_centered_energy_fraction'])[:5]:print(r)
if __name__=='__main__':main()
