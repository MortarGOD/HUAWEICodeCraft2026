"""Frozen cross-version audit. Does not optimize dataset mixture or inputs."""
import argparse
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution,evaluate_attention,gqa_attention,mse
from hif4_quant import dequantize_nvfp4,dequantize_hif4,quantize_to_hif4
VERSIONS=(285,302,310,311,324,365,381,382,386,387)
# Recent held-out mechanisms were not used to pick generator parameters.
# Differences below are total-score directions, NOT known exact Attention splits.
EDGES=[(285,302,'historical',1659),(310,311,'historical',-4995),
       (324,365,'historical',230.25),(365,381,'heldout',65),
       (381,382,'heldout',60),(382,386,'heldout',-160),(382,387,'heldout',25)]


def features(g):
    s=g['test'][0];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
    raw={r:dequantize_nvfp4(*s[r]).float() for r in 'qkv'}
    b={r:dequantize_hif4(quantize_to_hif4(x),x.shape).float() for r,x in raw.items()}
    n=raw['q'].shape[0];ix=torch.linspace(0,n-1,min(n,32)).long()
    q=raw['q'][ix].reshape(-1,qh,d).transpose(0,1)
    k=raw['k'].reshape(n,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
    v=raw['v'].reshape(n,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
    qb=b['q'][ix].reshape(-1,qh,d).transpose(0,1)
    kb=b['k'].reshape(n,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
    vb=b['v'].reshape(n,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
    p=torch.softmax(q@k.transpose(1,2)/d**.5,-1)
    pb=torch.softmax(qb@kb.transpose(1,2)/d**.5,-1)
    e1=(pb-p)@v;e2=p@(vb-v);e3=(pb-p)@(vb-v)
    total=(e1+e2+e3).square().mean()
    parts=[float(x.square().mean()) for x in (e1,e2,e3)]
    return dict(tokens=n,qh=qh,kh=kh,d=d,mean_max_p=float(p.max(-1).values.mean()),
        effective_tokens=float((1/p.square().sum(-1)).mean()),entropy=float(-(p*p.clamp_min(1e-30).log()).sum(-1).mean()),
        kv_equal=all(torch.equal(a,b) for a,b in zip(s['k'],s['v'])),
        baseline_mse=float(total),qk_term=parts[0],v_term=parts[1],interaction_term=parts[2],
        cross_terms=float(total)-sum(parts),identity_error=float(((pb@vb-p@v)-(e1+e2+e3)).abs().max()))


def summarize(report):
    assessment={}
    for bank,rows in report['banks'].items():
        assessment[bank]={}
        for a,b,split,target in EDGES:
            used=[r for r in rows if str(a) in r['versions'] and str(b) in r['versions']]
            if not used:continue
            delta=np.array([np.mean(r['versions'][str(b)]['scores'])-np.mean(r['versions'][str(a)]['scores']) for r in used])
            rng=np.random.default_rng(2026090602)
            boots=delta[rng.integers(0,len(delta),(2000,len(delta)))].mean(1)
            lo,hi=np.quantile(boots,[.025,.975])
            assessment[bank][f'{a}->{b}']=dict(split=split,online_total_delta=target,
                local_mean_delta=float(delta.mean()),direction_match=bool(delta.mean()*target>0),
                conditional_group_bootstrap=[float(lo),float(hi)],groups=len(delta),
                caveat='Bootstrap measures this proxy sample, not confidence about hidden online distribution.')
    report['assessment']=assessment
    report['score_prediction_allowed']=False


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='docs/structural-bank-audit.json')
    p.add_argument('--banks',default='old,new');p.add_argument('--versions',default=','.join(map(str,VERSIONS)))
    archive='local_fitted_archive/attention_feedback_v2_before_structural_20260906'
    active='local_fitted_models_attention_feedback_v2'
    candidate='local_fitted_models_structural_v3_candidate'
    p.add_argument('--old-bank',default=archive if (ROOT/archive).exists() else active)
    p.add_argument('--new-bank',default=candidate if (ROOT/candidate).exists() else active)
    args=p.parse_args();torch.set_num_threads(4)
    paths={'old':args.old_bank,'new':args.new_bank}
    ids={'old':[0,4,6,10,13,18,24,29,34,40],'new':list(range(10))}
    report=dict(complete=False,parameters_frozen=True,scope='10 groups / 50 cases per bank, not full population',
        banks={},source_hashes={str(v):hashlib.sha256((ROOT/f'solution/solution_v{v}.py').read_bytes()).hexdigest() for v in map(int,args.versions.split(','))})
    for name in args.banks.split(','):
        bank=torch.load(ROOT/paths[name]/'attn.pt',weights_only=False)
        report['banks'][name]=[]
        for gi in ids[name]:
            g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
            row=dict(group=gi,key=g['key'],features=features(g),versions={})
            report['banks'][name].append(row)
            common=None
            for version in map(int,args.versions.split(',')):
                m=load_solution(ROOT/f'solution/solution_v{version}.py')
                if version in (365,381,382,386,387):
                    if common is None:
                        parent=load_solution(ROOT/'solution/solution_v387.py')
                        common=parent._vm_parent_calibration(g['calib'],qh,kh,d)
                    m._temp_parent_calibration=lambda *a:copy.deepcopy(common)
                    if version in (382,386,387):m._vm_parent_calibration=lambda *a:copy.deepcopy(common)
                with contextlib.redirect_stdout(io.StringIO()):scores,seconds=evaluate_attention(m,[g])
                row['versions'][str(version)]=dict(scores=scores,seconds=seconds,
                    oj=dict(getattr(m,'OJ_DEBUG',{})),co=dict(getattr(m,'CO_DEBUG',{})),lt=dict(getattr(m,'LT_DEBUG',{})))
                print(name,gi,version,round(sum(scores),6),flush=True)
                summarize(report)
                (ROOT/args.output).write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True;summarize(report)
    (ROOT/args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
