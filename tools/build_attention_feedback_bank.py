"""Materialize a frozen population fit and test on independently generated groups.

Never overwrites old banks. Copies the frozen Linear dataset unchanged. A fit
that fails acceptance remains a diagnostic dataset, never a score predictor.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from build_model_proxy_dataset import MODELS,build_attention_group
from probe_attention_population import PROFILES,transform_group
from probe_persistent_covariance import run
from fit_attention_feedback import EDGES,absolute_score_audit
from self_check import _normalize_attention_dataset,_normalize_linear_dataset


def apportion(weights, count):
    w=np.asarray(weights,dtype=float)
    if not np.isfinite(w).all() or (w<0).any() or abs(w.sum()-1)>1e-6:
        raise ValueError('weights must be finite, nonnegative, and sum to one')
    exact=w*count
    result=np.floor(exact).astype(int)
    for i in np.argsort(-(exact-result),kind='stable')[:count-int(result.sum())]:
        result[i]+=1
    return result


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fit-report',required=True)
    p.add_argument('--output',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--report',default='docs/attention-feedback-bank-validation.json')
    p.add_argument('--seed',type=int,default=202609143)
    p.add_argument('--groups',type=int,default=50)
    p.add_argument('--versions',default='301,302,303,305,306,307,309,310,311,314')
    a=p.parse_args()
    fit=json.loads((ROOT/a.fit_report).read_text())
    dest=ROOT/a.output
    if dest.exists():p.error('choose a new output directory; overwriting banks is prohibited')
    if a.groups<1:p.error('--groups must be positive')
    if a.seed==fit['seed']:p.error('validation seed must differ from fitting seed')
    ids=[r['id'] for r in fit['records']]
    weights=[fit['fit']['weights'][i] for i in ids]
    counts=apportion(weights,a.groups)
    groups=[];assignments=[]
    rows=tuple(fit['rows'])
    for ci,(r,count) in enumerate(zip(fit['records'],counts)):
        for gi in range(int(count)):
            seed=a.seed+ci*100000+gi*37
            model=MODELS[r['model_index']]
            g=build_attention_group(model,seed,'covariance',rows,
                'checkpoint-projection',ROOT/'local_model_tensor_cache',0.05,128,4.)
            g=transform_group(g,PROFILES[r['profile']],seed+900000)
            # Attention lengths are a modeled hypothesis, not Linear strides.
            g.pop('logical_strides',None)
            g['modeled_token_counts']=list(rows)
            g['key']+=f'/feedback/{r["profile"]}'
            groups.append(g)
            assignments.append({'key':g['key'],'component':r['id'],'seed':seed})
        if count:print(r['id'],'groups',int(count),flush=True)
    _normalize_attention_dataset(groups)
    linear_source=ROOT/'local_fitted_models_covariance_v1/linear.pt'
    _normalize_linear_dataset(torch.load(linear_source,weights_only=False))
    dest.mkdir(parents=True)
    torch.save(groups,dest/'attn.pt')
    shutil.copy2(linear_source,dest/'linear.pt')
    linear_hash=sha256(linear_source)
    if sha256(dest/'linear.pt')!=linear_hash:raise RuntimeError('Linear copy differs')
    manifest={'status':'diagnostic; validation pending','score_prediction_allowed':False,
        'source_fit':a.fit_report,'source_fit_sha256':sha256(ROOT/a.fit_report),
        'source_fit_qualified':fit['fit']['qualified'],'seed':a.seed,
        'attention_groups':len(groups),'attention_cases':sum(len(g['test']) for g in groups),
        'modeled_token_counts':list(rows),'weights':dict(zip(ids,weights)),
        'realized_counts':{i:int(c) for i,c in zip(ids,counts)},'assignments':assignments,
        'linear_sha256':linear_hash,'linear_source':str(linear_source),
        'provenance':'real checkpoint projections + modeled hidden states and token-sink stresses; NOT captured model activations',
        'limitations':['mixture weights are non-identifiable, not true model proportions',
            'post-projection perturbations need not preserve DeepSeek K=V identity',
            'Attention lengths are fitted hypotheses, not known online lengths',
            'calibration/test samples are independent draws with shared population covariance',
            'local hardware/heads/shapes cannot predict 96-core online runtime']}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report={'manifest':manifest,'results':{}}
    for version in map(int,a.versions.split(',')):
        report['results'].update(run(groups,[version]))
        (ROOT/a.report).write_text(json.dumps(report,indent=2)+'\n')
        r=report['results'][str(version)]
        print('version',version,'raw',r['score'],'seconds',r['seconds'],flush=True)
    assessment={}
    for edge,meta in EDGES.items():
        left,right=edge.split('->')
        if left not in report['results'] or right not in report['results']:continue
        delta=np.asarray(report['results'][right]['cases'])-report['results'][left]['cases']
        predicted=25000*float(delta.mean())
        # The sampling unit is a group, not its five correlated test cases.
        group_delta=delta.reshape(len(groups),5).mean(1)
        rng=np.random.default_rng(a.seed)
        # Preserve the fitted component counts when resampling groups.
        boot=np.zeros(2000)
        offset=0
        for count in counts:
            count=int(count)
            if count:
                part=group_delta[offset:offset+count]
                boot+=rng.choice(part,size=(2000,count),replace=True).sum(1)
                offset+=count
        interval=np.quantile(25000*boot/len(groups),[.025,.975]).tolist()
        assessment[edge]={**meta,'predicted':predicted,'residual':predicted-meta['target'],
            'conditional_group_bootstrap_95':interval,
            'pass':abs(predicted-meta['target'])<=meta['tolerance']}
    absolute=absolute_score_audit({v:r['cases'] for v,r in report['results'].items()})
    qualified=(fit['fit']['qualified'] and len(assessment)==len(EDGES)
               and all(e['pass'] for e in assessment.values())
               and all(e['pass'] for e in absolute.values()))
    manifest['status']='qualified behavioral proxy' if qualified else 'diagnostic only; acceptance failed'
    manifest['score_prediction_allowed']=qualified
    report['validation_edges']=assessment
    report['absolute_score_audit']=absolute
    report['absolute_score_convention']='Conditional on 100 points per ratio unit per case and unweighted sum; verify if hidden scoring differs.'
    report['qualified']=qualified
    report['bootstrap_warning']='Conditional sampling variability only; excludes model/distribution uncertainty.'
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/a.report).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(assessment,indent=2),flush=True)
    print('qualified',qualified,flush=True)


if __name__=='__main__':main()
