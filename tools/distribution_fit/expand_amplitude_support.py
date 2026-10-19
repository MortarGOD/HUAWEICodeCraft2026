"""Extend a verified shared-mixture seed with fully measured, zero-weight columns."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'docs/distribution-fit'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--source',default='amplitude-constraint-growth-round2.json');parser.add_argument('--output',default='amplitude-expanded-start.json');args=parser.parse_args()
 source=D/args.source;p=json.loads(source.read_text());targets=json.loads((D/'targets.json').read_text());versions=targets['versions'];columns=[]
 for manifest in sorted((ROOT/'local_distribution_fit').glob('round*/atoms.json')):
  rnd=manifest.parent.name
  for atom in json.loads(manifest.read_text())['atoms']:
   if not atom['fit_eligible']:continue
   signature=None;complete=True
   for v in versions:
    f=D/rnd/'cells'/f'v{v["version"]}--{atom["id"]}.json'
    if not f.exists():complete=False;break
    c=json.loads(f.read_text())
    if not c['complete'] or c['source_sha256']!=v['source_sha256'] or c['data_sha256']!=atom['sha256']:complete=False;break
    this=[(x['test'],x['tokens']) for x in c['cases']]
    assert signature is None or signature==this
    signature=this
   if complete:
    columns.extend(rnd+'/'+atom['id']+f'/test{test}-tokens{tokens}' for test,tokens in signature)
 assert set(p['weights']).issubset(columns), 'Parent support lost valid complete measurements'
 added=[c for c in columns if c not in p['weights']]
 assert len(columns)==len(set(columns))
 for c in added:p['weights'][c]=0.
 p['support_extension']={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'complete_versions_required':len(versions),'complete_columns':len(columns),'added_columns':added,'limits':'Initial predictions unchanged because added weights are zero. New columns become useful only after a fresh joint solve and independent verification.'}
 (D/args.output).write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps(p['support_extension']))
if __name__=='__main__':main()
