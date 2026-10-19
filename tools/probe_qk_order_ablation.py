"""Remove separable terminal corrections, retaining paired transforms/Hessians."""
import contextlib,copy,io,json
import torch
from probe_v319_value import ROOT,load_solution
from probe_linear_deployed_joint import tree_equal
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result


def main():
    torch.set_num_threads(4)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    m=load_solution(ROOT/'solution/solution_v324.py');cal=m.hif4_calibration_attention;rows=[]
    for gi in (49,14,19):
        g=bank[gi];dims=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        parent=cal(g['calib'],*dims);snapshot=copy.deepcopy(parent);results={}
        for label in ('baseline','no_temperature','no_translation','neither'):
            state=copy.deepcopy(parent)
            if label in ('no_temperature','neither'):state['q_state'].pop('temperature_gain',None)
            if label in ('no_translation','neither'):
                state['k_state']={k:v for k,v in state['k_state'].items() if not k.startswith(('kt_','ts_'))}
            assert tree_equal(state['v_state'],parent['v_state'])
            for role in ('q_state','k_state'):
                for key in ('joint_transform','qs','ks','rot','H','u'):
                    assert tree_equal(state[role].get(key),parent[role].get(key))
            assert not validate_attention_calibration_result(state,'ablation')
            m.hif4_calibration_attention=lambda *_,_st=state:copy.deepcopy(_st)
            with contextlib.redirect_stdout(io.StringIO()):scores,_=evaluate_attention(m,[g])
            results[label]=scores
        assert tree_equal(parent,snapshot)
        row={'group':gi,'scores':results,'deltas':{k:sum(v)-sum(results['baseline']) for k,v in results.items()}}
        rows.append(row);print(row,flush=True)
    (ROOT/'docs/qk-order-ablation.json').write_text(json.dumps({'complete':True,'rows':rows,
        'warning':'Offline removal diagnostic, not a deployable improvement or online score estimate.'},indent=2)+'\n')


if __name__=='__main__':main()
