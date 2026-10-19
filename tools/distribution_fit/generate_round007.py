"""Target remaining support gaps through input gain and calibration/test drift."""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from tools.probe_attention_population import transform_group
DEST=ROOT/'local_distribution_fit/round007'
def main():
 torch.set_num_threads(2)
 if (DEST/'atoms.json').exists():raise SystemExit('round already exists')
 DEST.mkdir(parents=True,exist_ok=True);records=[]
 specs=[('trace35-group7',ROOT/'local_distribution_fit/round003/trace35-group7.pt',gain,False) for gain in (.75,1.25,2.)]
 specs += [('sink-drift',ROOT/'local_distribution_fit/round001/m0-sink.pt',gain,True) for gain in (.5,2.)]
 for i,(label,source,gain,drift) in enumerate(specs):
  g=torch.load(source,weights_only=True);seed=2026091400+i
  transformed=transform_group(g,{'gain':gain},seed)
  if drift:transformed['calib']=g['calib']
  name=f'{label}-gain{gain}';out=DEST/(name+'.pt');torch.save(transformed,out)
  records.append({'id':name,'path':str(out.relative_to(ROOT)),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'seed':seed,'mode':'block_resample' if drift else 'row_permute','fit_eligible':True,'source_path':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'profile':{'gain':gain,'test_only':drift}})
 report={'round':7,'atoms':records,'selection_evidence':'v374 max +206.59 vs target +226 on trace35/group7; v396 available differences nonnegative. Vary actual input gain and test drift without modifying candidate gates or scoring.','limits':'Adaptive synthetic distribution hypotheses; no hidden-distribution claim.'}
 text=json.dumps(report,indent=2)+'\n';(DEST/'atoms.json').write_text(text);doc=ROOT/'docs/distribution-fit/round007';doc.mkdir(exist_ok=True);(doc/'atoms.json').write_text(text)
if __name__=='__main__':main()
