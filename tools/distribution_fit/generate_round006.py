"""Vary fake-peer correlation while keeping true calibration/test tensors fixed."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DEST=ROOT/'local_distribution_fit/round006'
def main():
 if (DEST/'atoms.json').exists():raise SystemExit('round already exists')
 DEST.mkdir(parents=True,exist_ok=True);records=[]
 for source_round,source_id in [('round003','trace25-group1-correlated_noise'),('round005','rank8-strength4-length128')]:
  source=next(a for a in json.loads((ROOT/'local_distribution_fit'/source_round/'atoms.json').read_text())['atoms'] if a['id']==source_id)
  for noise in (.005,.01):
   records.append(dict(source,id=source_id+f'-noise{noise}',mode='correlated_noise',noise=noise,fit_eligible=True))
 report={'round':6,'atoms':records,'selection_evidence':'Remaining v330/v374/v396 range gaps after covariance screen. Test whether independently calibrated transforms and acceptance gates are sensitive to smaller fake-peer perturbations.','limits':'High fake-peer correlation is an unverified hypothesis. Noise is nonzero but NVFP4 rounding may erase some perturbations; not evidence of the hidden protocol.'}
 text=json.dumps(report,indent=2)+'\n';(DEST/'atoms.json').write_text(text);doc=ROOT/'docs/distribution-fit/round006';doc.mkdir(exist_ok=True);(doc/'atoms.json').write_text(text)
if __name__=='__main__':main()
