"""Allow independent decoy protocols for the three independent calibrations."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DEST=ROOT/'local_distribution_fit/round009'
def main():
 if (DEST/'atoms.json').exists():raise SystemExit('round already exists')
 DEST.mkdir(parents=True,exist_ok=True);records=[]
 sources=[('round001','m0-base-block_resample'),('round001','m0-sink-block_resample'),('round003','trace25-group1-correlated_noise'),('round005','rank8-strength4-length128')]
 protocols={'qk-close-v-block':dict(q='correlated_noise',k='correlated_noise',v='block_resample'),'qk-block-v-close':dict(q='block_resample',k='block_resample',v='correlated_noise'),'q-row-k-close-v-block':dict(q='row_permute',k='correlated_noise',v='block_resample')}
 for round_name,source_id in sources:
  a=next(x for x in json.loads((ROOT/'local_distribution_fit'/round_name/'atoms.json').read_text())['atoms'] if x['id']==source_id)
  for name,roles in protocols.items():records.append(dict(a,id=source_id+'-'+name,role_modes=roles,noise=.05,fit_eligible=True))
 r={'round':9,'atoms':records,'selection_evidence':'90-column separation couples Query and Value variants. Prior banks applied one common decoy protocol to all three independent calibration roles; relax this unsupported equality while preserving own-role-only state.','limits':'Role-specific fake-peer distributions are hypotheses. No candidate code, target receipt, tolerance, or dynamic information boundary changes.'}
 text=json.dumps(r,indent=2)+'\n';(DEST/'atoms.json').write_text(text);doc=ROOT/'docs/distribution-fit/round009';doc.mkdir(exist_ok=True);(doc/'atoms.json').write_text(text)
if __name__=='__main__':main()
