"""Check budget pruning against both candidates on exactly shared parents."""
import copy
import json
import torch
from probe_v319_value import ROOT,load_solution
from build_v321_candidate import candidate_source
from test_v320_residual import same


def main():
    torch.set_num_threads(4);m=load_solution(ROOT/'solution/solution_v320.py')
    exec(compile(candidate_source(),'budget-guard-test','exec'),m.__dict__)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    results=[]
    for gi in (0,1,11,14,19,47,48,49):
        g=bank[gi];args=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        parent=m._kt_parent_calibration(g['calib'],*args);snapshot=copy.deepcopy(parent)
        m._KT_BUDGET_ROUTE=False;m._KT_GUARD_ROWS=64;m._KT_MIN_GAIN=.005
        full=m._kt_guard(g['calib'],parent,*args);old=copy.deepcopy(m.KT_DEBUG)
        m._KT_BUDGET_ROUTE=True;m._KT_GUARD_ROWS=32;m._KT_MIN_GAIN=.01
        pruned=m._kt_guard(g['calib'],parent,*args);new=copy.deepcopy(m.KT_DEBUG)
        assert same(parent,snapshot)
        results.append(dict(group=gi,refinement=int(parent['k_state'].get('rf') or 0),
            states_equal=same(full,pruned),full_chosen=old.get('chosen',[]),pruned_chosen=new.get('chosen',[]),
            old_candidates=len(old.get('modes',[])),new_candidates=len(new.get('modes',[])),
            guard_rows=new.get('guard_rows'),minimum_gain=new.get('minimum_gain')))
    report=dict(complete=True,parent_unchanged=True,all_states_equal=all(r['states_equal'] for r in results),groups=results,
                warning='Empirical calibration equality on listed groups, not a proof of global pruning equivalence.')
    (ROOT/'docs/v321-budget-route-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(report)


if __name__=='__main__':main()
