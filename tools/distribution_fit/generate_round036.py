"""Enumerate all nonidentity test V permutations on a fixed four-token source."""
import copy,hashlib,itertools,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
def main():
 torch.set_num_threads(2);dest=ROOT/'local_distribution_fit/round036';assert not dest.exists()
 source=next(a for a in json.loads((ROOT/'local_distribution_fit/round003/atoms.json').read_text())['atoms'] if a['id']=='trace25-group40-row_permute')
 path=ROOT/source['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256'];g=torch.load(path,weights_only=True)
 assert all(s['v'][0].shape[0]==4 for s in g['test'])
 dest.mkdir();records=[]
 for order in itertools.permutations(range(4)):
  if order==(0,1,2,3):continue
  current=copy.deepcopy(g);ix=torch.tensor(order)
  for original,s in zip(g['test'],current['test']):
   s['v']=[x[ix].contiguous() for x in original['v']]
   for a,b in zip(original['v'],s['v']):assert torch.equal(a[ix],b)
   for r in 'qk':
    for a,b in zip(original[r],s[r]):assert torch.equal(a,b)
  for a,b in zip(g['calib'],current['calib']):
   for r in 'qkv':
    for x,y in zip(a[r],b[r]):assert torch.equal(x,y)
  name='perm'+''.join(map(str,order));out=dest/(name+'.pt');torch.save(current,out)
  records.append(dict(source,id=name,path=str(out.relative_to(ROOT)),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),source_path=source['path'],source_sha256=source['sha256'],profile={'test_v_row_permutation':order,'calibration_unchanged':True}))
 assert len(records)==23
 prices=json.loads((ROOT/'docs/distribution-fit/amplitude-separator-prices.json').read_text())
 report={'round':36,'atoms':records,'certificate_sha256':prices['certificate_sha256'],'required_versions':prices['required_versions'],'selection_evidence':'Enumerate all 23 nonidentity permutations on current closest four-token source. Identity already measured. No candidate errors or outputs choose the order.','limits':'Exhaustive only for fixed-source test V row permutations, not all datasets or calibration protocols. Positive separator margin still requires complete measurement and joint verification.'}
 doc=ROOT/'docs/distribution-fit/round036';doc.mkdir();text=json.dumps(report,indent=2)+'\n';(dest/'atoms.json').write_text(text);(doc/'atoms.json').write_text(text);print('23 immutable permutations generated and checked')
if __name__=='__main__':main()
