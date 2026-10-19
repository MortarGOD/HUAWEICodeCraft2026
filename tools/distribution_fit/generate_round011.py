"""Certificate-directed neighborhood search over decoy permutation seeds."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];DEST=ROOT/'local_distribution_fit/round011'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--round',type=int,default=11);args=parser.parse_args()
 if args.round<11:raise SystemExit('adaptive seed rounds must be >= 11')
 DEST=ROOT/f'local_distribution_fit/round{args.round:03d}'
 if (DEST/'atoms.json').exists():raise SystemExit('round already exists')
 prices=json.loads((ROOT/'docs/distribution-fit/conflict-prices.json').read_text());best=prices['ranked_cases'][0];rnd,name=best['population'].split('/',1)
 source=next(a for a in json.loads((ROOT/'local_distribution_fit'/rnd/'atoms.json').read_text())['atoms'] if a['id']==name)
 assert source['mode'] in ('row_permute','block_resample')
 records=[dict(source,id=f'{name}-seedplus{delta}',seed=source['seed']+delta) for delta in (1,2,3,4)]
 report={'round':args.round,'atoms':records,'selection_evidence':best,'required_versions':prices['required_versions'],'limits':'Adaptive supervised support search using certificate prices. True tensors unchanged; all versions share each fixed decoy seed. Breaking one certificate is not final feasibility.'}
 DEST.mkdir(parents=True,exist_ok=True);text=json.dumps(report,indent=2)+'\n';(DEST/'atoms.json').write_text(text);doc=ROOT/f'docs/distribution-fit/round{args.round:03d}';doc.mkdir(exist_ok=True);(doc/'atoms.json').write_text(text)
if __name__=='__main__':main()
