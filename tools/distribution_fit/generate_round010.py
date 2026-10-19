"""Change only fake Q/K amplitude in V calibration to test probability mismatch."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DEST=ROOT/'local_distribution_fit/round010'
def main():
 if (DEST/'atoms.json').exists():raise SystemExit('round already exists')
 DEST.mkdir(parents=True,exist_ok=True);records=[]
 for rnd,atom in [('round005','rank8-strength4-length128'),('round003','trace25-group1-correlated_noise')]:
  source=next(a for a in json.loads((ROOT/'local_distribution_fit'/rnd/'atoms.json').read_text())['atoms'] if a['id']==atom)
  for gain in (.25,.5,2.):records.append(dict(source,id=atom+f'-v-decoy-gain{gain}',decoy_gains={'v':{'q':gain,'k':gain}}))
 report={'round':10,'atoms':records,'selection_evidence':'Mean-search acceptance depends on fake-Q/K probabilities during V calibration. Change their amplitude independently of true calibration roles and test inputs to investigate coupled Q/Value constraints.','limits':'Unverified fake-peer distribution hypothesis. Only non-target roles are rescaled; candidate code and score targets unchanged.'}
 text=json.dumps(report,indent=2)+'\n';(DEST/'atoms.json').write_text(text);doc=ROOT/'docs/distribution-fit/round010';doc.mkdir(exist_ok=True);(doc/'atoms.json').write_text(text)
if __name__=='__main__':main()
