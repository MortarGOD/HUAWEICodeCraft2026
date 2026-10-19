"""Paired evaluation; local scores are not online predictions."""
import argparse
import contextlib
import io
import json
import hashlib
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import load_solution, evaluate_attention, evaluate_linear
from self_check import validate_attention_calibration_result, validate_linear_calibration_result, validate_hif4_params


def instrument(module):
    stats={'checks':0,'calibration_s':0.,'dynamic_s':0.}
    depth=[0]
    for name in ('hif4_calibration_attention','hif4_calibration_and_quantize_weight'):
        original=getattr(module,name)
        def cal(*args,_fn=original,_name=name):
            depth[0]+=1;start=time.perf_counter()
            try: result=_fn(*args)
            finally:
                stats['calibration_s']+=time.perf_counter()-start;depth[0]-=1
            errors=(validate_attention_calibration_result(result,'attention') if _name.endswith('attention')
                else validate_linear_calibration_result(result,args[0].shape,'linear'))
            assert not errors,errors
            stats['checks']+=1
            return result
        setattr(module,name,cal)
    for role in ('activation','q','k','v'):
        original=getattr(module,'hif4_dynamic_quantize_'+role)
        def dyn(*args,_fn=original,_role=role):
            start=time.perf_counter();result=_fn(*args)
            if not depth[0]:
                stats['dynamic_s']+=time.perf_counter()-start
                errors=validate_hif4_params(result,args[0].shape,_role)
                assert not errors,errors
                stats['checks']+=1
            return result
        setattr(module,'hif4_dynamic_quantize_'+role,dyn)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--versions', default='365,366,367')
    ap.add_argument('--groups', default='0,4,14,19,49')
    ap.add_argument('--scenario', choices=['attention','linear'], default='attention')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    bank = torch.load(ROOT/'local_fitted_models_attention_feedback_v2'/('attn.pt' if args.scenario=='attention' else 'linear.pt'),weights_only=False)
    modules = {v:load_solution(ROOT/f'solution/solution_v{v}.py') for v in ['324']+args.versions.split(',')}
    stats={v:instrument(m) for v,m in modules.items()}
    hashes={v:hashlib.sha256((ROOT/f'solution/solution_v{v}.py').read_bytes()).hexdigest() for v in modules}
    rows=[]
    indices=range(len(bank)) if args.groups=='all' else list(map(int,args.groups.split(',')))
    for index in indices:
        group=bank[index]; row={'group':index,'versions':{}}
        order=list(modules)
        if index%2:order.reverse()
        for version in order:
            m=modules[version];start_stats=dict(stats[version])
            with torch.inference_mode(),contextlib.redirect_stdout(io.StringIO()):
                scores,elapsed=(evaluate_attention if args.scenario=='attention' else evaluate_linear)(m,[group])
            debug=next((getattr(m,key) for key in ('OJ_DEBUG','HS_DEBUG','NQ_DEBUG','DG_DEBUG','CM_DEBUG') if hasattr(m,key)),{})
            result={'scores':scores,'elapsed_s':elapsed,'debug':dict(debug)}
            result.update({key:stats[version][key]-start_stats[key] for key in start_stats})
            assert all(torch.isfinite(torch.tensor(scores)))
            row['versions'][version]=result
            print(index,version,sum(scores),elapsed,result['debug'],flush=True)
        rows.append(row)
        (ROOT/args.output).write_text(json.dumps({'complete':False,'scenario':args.scenario,'rows':rows,'sha256':hashes,'stats':stats,'online_prediction':None},indent=2)+'\n')
    (ROOT/args.output).write_text(json.dumps({'complete':True,'scenario':args.scenario,'rows':rows,'sha256':hashes,'stats':stats,'online_prediction':None},indent=2)+'\n')


if __name__=='__main__':
    main()
