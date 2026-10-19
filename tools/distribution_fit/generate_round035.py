"""Permute exact NVFP4 V rows by attention leverage, without quantizer outputs."""
import copy, hashlib, json, math, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from hif4_quant import dequantize_nvfp4

def main():
 torch.set_num_threads(2)
 dest=ROOT/'local_distribution_fit/round035'
 assert not dest.exists(), 'Immutable round already exists'
 source=next(a for a in json.loads((ROOT/'local_distribution_fit/round003/atoms.json').read_text())['atoms'] if a['id']=='trace25-group40-row_permute')
 path=ROOT/source['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
 original=torch.load(path,weights_only=True);qh,kh,d=[original[x] for x in ('q_num_heads','kv_num_heads','head_dim')]
 dest.mkdir();records=[];audit=[]
 for reverse in (False,True):
  for phases in (('test',),('calib','test')):
   group=copy.deepcopy(original);name=('anti' if reverse else 'aligned')+('-test' if len(phases)==1 else '-both');stats=[]
   for phase in phases:
    for j,s in enumerate(group[phase]):
     raw={r:dequantize_nvfp4(*s[r]).float() for r in 'qkv'};t=raw['v'].shape[0]
     q=raw['q'].reshape(t,qh,d).transpose(0,1);k=raw['k'].reshape(t,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
     prob=(q@k.transpose(1,2)/math.sqrt(d)).softmax(-1)
     leverage=prob.square().sum((0,1));energy=raw['v'].square().mean(-1)
     positions=torch.argsort(leverage,stable=True);values=torch.argsort(energy,descending=reverse,stable=True)
     order=torch.empty(t,dtype=torch.long);order[positions]=values
     s['v']=[x[order].contiguous() for x in s['v']]
     for before,after in zip(original[phase][j]['v'],s['v']):assert torch.equal(after,before[order])
     for r in 'qk':
      for a,b in zip(original[phase][j][r],s[r]):assert torch.equal(a,b)
     stats.append({'phase':phase,'test':j,'tokens':t,'moved_rows':int((order!=torch.arange(t)).sum()),'weighted_energy_before':float(leverage@energy),'weighted_energy_after':float(leverage@energy[order])})
   for phase in ('calib','test'):
    if phase not in phases:
     for a,b in zip(original[phase],group[phase]):
      for r in 'qkv':
       for x,y in zip(a[r],b[r]):assert torch.equal(x,y)
   out=dest/(name+'.pt');torch.save(group,out)
   records.append(dict(source,id=name,path=str(out.relative_to(ROOT)),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),source_path=source['path'],source_sha256=source['sha256'],profile={'v_energy_attention_leverage_order':'opposite' if reverse else 'same','phases':phases,'exact_nvfp4_row_permutation':True}))
   audit.append({'atom':name,'rows':stats})
 prices=json.loads((ROOT/'docs/distribution-fit/amplitude-separator-prices.json').read_text())
 report={'round':35,'atoms':records,'certificate_sha256':prices['certificate_sha256'],'required_versions':prices['required_versions'],'selection_evidence':'The current independently verified 565-column separator is closest to trace25 group40. Probe exact V row ordering on this distinct source with calibration/test controls; prior sink-source test-only ordering broke an old separator but did not improve joint fit. No candidate outputs or errors select rows.','limits':'Adaptive synthetic training distributions; no evidence this correlation holds online. Shared identical inputs and protocol across versions.'}
 doc=ROOT/'docs/distribution-fit/round035';doc.mkdir()
 text=json.dumps(report,indent=2)+'\n';(dest/'atoms.json').write_text(text);(doc/'atoms.json').write_text(text);(doc/'input-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 print('Generated four exact-row permutation populations')
if __name__=='__main__':main()
