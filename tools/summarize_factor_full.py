"""Describe full proxy regression without fitting weights or online scores."""
import json
from collections import defaultdict
import torch
from probe_v319_value import ROOT


def summarize(report,bank):
    assert report['complete'] and report['group_count']==len(bank)
    assert {r['group'] for r in report['rows']}==set(range(len(bank)))
    buckets=defaultdict(lambda:dict(cases=0,delta=0.,up=0,down=0,same=0))
    groups=[]
    for row in report['rows']:
        g=bank[row['group']]
        old=row['scores']['324'];new=row['scores']['350']
        assert len(old)==len(new)==len(g['test'])
        groups.append(dict(group=row['group'],key=g['key'],delta=row['delta']['350'],
                           hold_ratio=row['debug']['350'].get('hold_ratio'),
                           adopted=row['debug']['350'].get('adopted',0)))
        for s,a,b in zip(g['test'],old,new):
            delta=b-a
            for key in ('all','source:'+g['source_model'],'length:'+str(s['v'][0].shape[0]),
                        'screen' if row['group'] in {0,1,5,14,19,49} else 'outside_screen'):
                v=buckets[key];v['cases']+=1;v['delta']+=delta
                v['up' if delta>1e-12 else 'down' if delta< -1e-12 else 'same']+=1
    assert buckets['all']['cases']==report['test_count']
    assert abs(buckets['all']['delta']-report['delta']['350'])<1e-10
    ratio=report['totals']['350']/report['totals']['324']
    return dict(complete=True,buckets=dict(buckets),groups=sorted(groups,key=lambda x:x['delta']),
                timing_ratio=ratio,conditional_seconds_246=246*ratio,
                warning='Descriptive proxy statistics only; not real-forward or online score evidence.')


if __name__=='__main__':
    report=json.loads((ROOT/'docs/v350-production-regression-full.json').read_text())
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    result=summarize(report,bank)
    (ROOT/'docs/v350-full-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='groups'},indent=2),flush=True)
    print('negative groups',[g for g in result['groups'] if g['delta']<0],flush=True)
