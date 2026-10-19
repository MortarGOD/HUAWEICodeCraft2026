"""Materialize GP-proposed shared calibration/test gain populations."""
import hashlib
import json
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from tools.probe_attention_population import transform_group

def main():
    torch.set_num_threads(2)
    dest=ROOT/'local_distribution_fit/round026'
    assert not (dest/'atoms.json').exists(), 'Immutable round already exists'
    proposals=json.loads((ROOT/'docs/distribution-fit/gp-scale-proposals.json').read_text())
    assert proposals['certificate_sha256']==hashlib.sha256((ROOT/'docs/distribution-fit/reduced-conflict-certificate.json').read_bytes()).hexdigest()
    source=next(a for a in json.loads((ROOT/'local_distribution_fit/round013/atoms.json').read_text())['atoms'] if a['id']=='sink-input-seed2026091600-seedplus4')
    path=ROOT/source['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
    group=torch.load(path,weights_only=True);dest.mkdir(parents=True,exist_ok=True);atoms=[]
    for i,p in enumerate(proposals['proposals']):
        outgroup=dict(group)
        for phase in ('calib','test'):
            transformed=transform_group(group,{'gain':p[phase+'_gain']},2026091800)
            outgroup[phase]=transformed[phase]
            for original,current in zip(group[phase],outgroup[phase]):
                current['v']=original['v']
                assert all(torch.equal(a,b) for a,b in zip(original['v'],current['v']))
        profile={k:p[k] for k in ('calib_gain','test_gain')}
        profile['v_encoding_preserved']=True
        outgroup['population_profile']=profile
        name=f'gp-scale-{i+1}';out=dest/(name+'.pt');torch.save(outgroup,out)
        atoms.append(dict(source,id=name,path=str(out.relative_to(ROOT)),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),source_path=source['path'],source_sha256=source['sha256'],profile=profile))
    prices=json.loads((ROOT/'docs/distribution-fit/conflict-prices.json').read_text())
    report={'round':26,'atoms':atoms,'required_versions':prices['required_versions'],'proposal_model':proposals,'limits':'Learned proposals require actual evaluation. No predicted cells enter fitting.'}
    text=json.dumps(report,indent=2)+'\n';(dest/'atoms.json').write_text(text)
    doc=ROOT/'docs/distribution-fit/round026';doc.mkdir(exist_ok=True);(doc/'atoms.json').write_text(text)
    print('Four GP proposals materialized with fixed source and preserved V')

if __name__=='__main__':main()
