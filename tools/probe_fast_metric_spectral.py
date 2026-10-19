"""Check one-pass calibration state/output equivalence against v358 prototype."""
import contextlib
import copy
import io
import json
import time

import torch

from attention_metric_spectral_source import metric_spectral_source
from attention_metric_spectral_fast_source import fast_metric_spectral_source
from probe_v319_value import ROOT, load_solution
from evaluate_local import evaluate_attention
from probe_linear_deployed_joint import tree_equal
from self_check import validate_attention_calibration_result


def main():
    torch.set_num_threads(4)
    slow = load_solution(ROOT / 'solution/solution_v353.py')
    fast = load_solution(ROOT / 'solution/solution_v353.py')
    exec(compile(metric_spectral_source(), 'slow_metric_spectral', 'exec'), slow.__dict__)
    exec(compile(fast_metric_spectral_source(), 'fast_metric_spectral', 'exec'), fast.__dict__)
    rows=[];checks=0
    for dataset,groups in (
        ('local_fitted_models_attention_feedback_v2',(0,1,2,5,14,49)),
        ('local_fitted_models_covariance_v1',(0,1,2,25,26,49)),
    ):
        bank=torch.load(ROOT/dataset/'attn.pt',weights_only=False)
        for group_index in groups:
            group=bank[group_index];dims=[group[key] for key in ('q_num_heads','kv_num_heads','head_dim')]
            results={}
            for label,module in (('slow',slow),('fast',fast)):
                calibration=module.hif4_calibration_attention
                start=time.perf_counter();state=calibration(group['calib'],*dims)
                elapsed=time.perf_counter()-start
                assert not validate_attention_calibration_result(state,label);checks+=1
                module.hif4_calibration_attention=lambda *_,_state=state:copy.deepcopy(_state)
                with contextlib.redirect_stdout(io.StringIO()):scores,_=evaluate_attention(module,[group])
                module.hif4_calibration_attention=calibration
                results[label]={'state':state,'scores':scores,'calibration_seconds':elapsed}
            equal=tree_equal(results['slow']['state'],results['fast']['state'])
            max_delta=max(abs(a-b) for a,b in zip(results['slow']['scores'],results['fast']['scores']))
            assert max_delta<1e-10
            checks+=1
            row={'dataset':dataset,'group':group_index,'state_equal':equal,'max_score_delta':max_delta,
                 'calibration_seconds':{k:v['calibration_seconds'] for k,v in results.items()}}
            rows.append(row);print(row,flush=True)
    report={'complete':True,'checks':checks,'rows':rows,
            'state_equal_groups':sum(row['state_equal'] for row in rows),
            'slow_seconds':sum(row['calibration_seconds']['slow'] for row in rows),
            'fast_seconds':sum(row['calibration_seconds']['fast'] for row in rows),
            'warning':'Paired local 4-thread calibration; not online runtime evidence.'}
    (ROOT/'docs/fast-metric-spectral-equivalence.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report,flush=True)


if __name__=='__main__':main()
