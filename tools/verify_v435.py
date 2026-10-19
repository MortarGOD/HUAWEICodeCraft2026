"""Post-push v430/v435 comparisons, with independent V constraint checks."""
import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tools.verify_attention_campaign as campaign
from hif4_quant import dequantize_hif4


def stats(error, buckets):
    error = error.double()
    sse = error.square().sum(0)
    mean = error.sum(0).square()
    objective = (sse + mean) / error.shape[0]
    for group in range(4):
        chosen = error[buckets == group]
        if len(chosen):objective += chosen.sum(0).square() / len(chosen)
    return objective, sse, mean


original_instrument = campaign.instrument


def instrument(module):
    counts = original_instrument(module)
    record = {'calls':0,'seconds':0.,'eligible_calls':0,'changed_calls':0,
              'changed_values':0,'enabled_heads':0,'changed_channels':0,
              'cap_checks':0,'exceptions':[],'cases':[]}
    counts['_v435_refine'] = record
    fn = module._v435_refine
    def wrapped(x, params, enabled, kh, d):
        frozen = copy.deepcopy((x,params,enabled))
        start = time.perf_counter()
        try:result = fn(x,params,enabled,kh,d)
        except Exception as exc:
            record['exceptions'].append(repr(exc));raise
        record['seconds'] += time.perf_counter()-start
        record['calls'] += 1
        assert campaign.equal(frozen,(x,params,enabled))
        before = dequantize_hif4(params,x.shape).double()
        after = dequantize_hif4(result,x.shape).double()
        changed = after != before
        record['changed_calls'] += int(bool(changed.any()))
        record['changed_values'] += int(changed.sum())
        record['enabled_heads'] += int(enabled.sum())
        eligible = 2<=x.shape[0]<=4096 and x.numel()<=2**21 and d%64==0
        record['eligible_calls'] += int(eligible)
        row = {'tokens':len(x),'kh':kh,'d':d,'eligible':eligible,
               'enabled_heads':int(enabled.sum()),'changed_values':int(changed.sum())}
        record['cases'].append(row)
        for head in range(kh):
            lo,hi=head*d,(head+1)*d
            if not enabled[head]:
                assert torch.equal(before[:,lo:hi],after[:,lo:hi]);continue
            if not eligible:continue
            buckets = module._v435_buckets(x[:,lo:hi])
            a = stats(before[:,lo:hi]-x[:,lo:hi].double(),buckets)
            b = stats(after[:,lo:hi]-x[:,lo:hi].double(),buckets)
            for old,new in zip(a,b):assert bool((new<=old).all()),(head,row)
            count = int(changed[:,lo:hi].any(0).sum())
            record['changed_channels'] += count
            record['cap_checks'] += d
        return result
    module._v435_refine = wrapped
    return counts


campaign.instrument = instrument

if __name__ == '__main__':
    jobs = [
        ('paired','local_fitted_models_attention_feedback_v2','0,8,17,25,34,49',False,'row_permute'),
        ('covariance','local_fitted_models_covariance_v1','0,8,17,25,34,49',False,'row_permute'),
        ('row_permute','local_fitted_models_attention_feedback_v2','0,24,49',True,'row_permute'),
        ('block_resample','local_fitted_models_attention_feedback_v2','0,24,49',True,'block_resample'),
    ]
    for name,dataset,groups,isolated,decoy in jobs:
        campaign.run(SimpleNamespace(parent=430,candidate=435,dataset=dataset,groups=groups,
                         isolated=isolated,decoy_mode=decoy,output=f'docs/v435-attention-{name}.json'))
    print('All four predeclared v435 comparisons completed.',flush=True)
