"""Replay version ablations on a fixed bank; never fit version-specific data.

Raw scores are unitless MSE improvements, NOT online weighted points.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import evaluate_linear
from probe_persistent_covariance import load, run, stats

ATTENTION_EDGES=[(1,6,864),(285,302,1830),(302,303,-2333),(303,305,184),
                 (305,306,331),(306,307,1819),(307,308,0)]
LINEAR_EDGES=[(10,11,1810),(28,29,3888),(31,32,2228),(229,239,5551),(254,255,423),
              (269,270,204),
              (301,285,171),(303,304,12),(307,308,0)]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--datasets-dir',required=True)
    p.add_argument('--component',choices=('linear','attention'),required=True)
    p.add_argument('--versions',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    field='attn.pt' if args.component=='attention' else 'linear.pt'
    groups=torch.load(Path(args.datasets_dir)/field,weights_only=False)
    record={'dataset':args.datasets_dir,'component':args.component,'versions':{},
            'score_units':'raw MSE improvement sums, not online weighted scores',
            'warning':'old linear edges have different online eras; direction diagnostics only'}
    if args.component=='attention': record['covariance_stats']=[stats(g) for g in groups]
    for version in map(int,args.versions.split(',')):
        torch.manual_seed(20260903)
        if args.component=='attention':
            result=run(groups,[version])[str(version)]
        else:
            m=load(version)
            with contextlib.redirect_stdout(io.StringIO()):
                scores,seconds=evaluate_linear(m,groups)
            result={'score':sum(scores),'seconds':seconds,'cases':scores}
        record['versions'][str(version)]=result
        edges=ATTENTION_EDGES if args.component=='attention' else LINEAR_EDGES
        record['edges']=[]
        for a,b,online in edges:
            if str(a) not in record['versions'] or str(b) not in record['versions']: continue
            delta=record['versions'][str(b)]['score']-record['versions'][str(a)]['score']
            record['edges'].append({'from':a,'to':b,'online_delta_reference':online,
                'local_delta':delta,'direction_pass':abs(delta)<1e-9 if online==0 else delta*online>0})
        Path(args.output).write_text(json.dumps(record,indent=2)+'\n')
        print(version,round(result['score'],6),round(result['seconds'],2),
              result.get('flags',{}),flush=True)
    print('EDGES',record['edges'],flush=True)


if __name__=='__main__': main()
