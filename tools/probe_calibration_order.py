"""Offline full-data order sensitivity; never choose an order from test scores."""
import contextlib
import copy
import io
import json
import torch
from probe_v319_value import ROOT,load_solution
from probe_linear_deployed_joint import tree_equal
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result


def main():
    torch.set_num_threads(4)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    module=load_solution(ROOT/'solution/solution_v324.py')
    calibration=module.hif4_calibration_attention
    orders={'identity':[0,1,2,3,4],'swap_last':[0,1,2,4,3],'reverse':[4,3,2,1,0]}
    rows=[]
    for gi in (0,1,5,14,19,49):
        group=bank[gi];dims=[group[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        results={};baseline=None
        for label,indices in orders.items():
            assert sorted(indices)==list(range(len(group['calib'])))
            state=calibration([group['calib'][i] for i in indices],*dims)
            assert not validate_attention_calibration_result(state,'order')
            same={role:False for role in ('q_state','k_state','v_state')}
            if baseline is None:baseline=copy.deepcopy(state)
            for role in same:same[role]=tree_equal(state[role],baseline[role])
            module.hif4_calibration_attention=lambda *_,_st=state:copy.deepcopy(_st)
            with contextlib.redirect_stdout(io.StringIO()):scores,_=evaluate_attention(module,[group])
            results[label]={'scores':scores,'same_state_as_identity':same}
        row={'group':gi,'results':results};rows.append(row);print(row,flush=True)
    totals={label:sum(sum(r['results'][label]['scores'])-sum(r['results']['identity']['scores']) for r in rows) for label in orders}
    report={'complete':True,'orders':orders,'rows':rows,'deltas':totals,
            'warning':'All five calibration samples retained. Modeled activations, no online inference; diagnostic only, no test-selected deployment.'}
    (ROOT/'docs/calibration-order-screen.json').write_text(json.dumps(report,indent=2)+'\n')
    print(totals,flush=True)


if __name__=='__main__':main()
