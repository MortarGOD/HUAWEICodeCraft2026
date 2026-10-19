"""Change test V/token correspondence while preserving calibration and decoy seeds."""
import copy,hashlib,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];DEST=ROOT/'local_distribution_fit/round008'
def main():
 torch.set_num_threads(2)
 if (DEST/'atoms.json').exists():raise SystemExit('round already exists')
 DEST.mkdir(parents=True,exist_ok=True)
 source=next(a for a in json.loads((ROOT/'local_distribution_fit/round001/atoms.json').read_text())['atoms'] if a['id']=='m0-sink-block_resample')
 path=ROOT/source['path'];group=torch.load(path,weights_only=True);records=[]
 for kind in ('random','reverse'):
  g=copy.deepcopy(group);generator=torch.Generator().manual_seed(2026091500)
  for sample in g['test']:
   rows=sample['v'][0].shape[0]
   order=torch.randperm(rows,generator=generator) if kind=='random' else torch.arange(rows-1,-1,-1)
   sample['v']=[tensor[order].contiguous() for tensor in sample['v']]
  name='v-token-'+kind;out=DEST/(name+'.pt');torch.save(g,out)
  records.append(dict(source,id=name,path=str(out.relative_to(ROOT)),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),source_path=str(path.relative_to(ROOT)),source_sha256=source['sha256'],profile={'test_v_token_permutation':kind,'calibration_unchanged':True,'permutation_seed':2026091500}))
 report={'round':8,'atoms':records,'selection_evidence':'v396 uses calibration K/V bank; original sink population has positive delta only. Preserve accepted calibration and test transfer under changed V/token correspondence.','limits':'Synthetic distribution shift; no claim about hidden correlations. Uses valid NVFP4 row permutations and unchanged scoring.'}
 text=json.dumps(report,indent=2)+'\n';(DEST/'atoms.json').write_text(text);doc=ROOT/'docs/distribution-fit/round008';doc.mkdir(exist_ok=True);(doc/'atoms.json').write_text(text)
if __name__=='__main__':main()
