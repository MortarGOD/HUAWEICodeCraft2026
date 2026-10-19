"""Actual standalone versus original research implementation on shared real-bank state."""
import copy
import json
import time
import torch
from build_attention_online_probes import ROOT,MODES
from probe_v319_value import load_solution
from probe_linear_deployed_joint import tree_equal
from probe_v326_regroup import install,transforms,metric_states,nonorthogonal_states
from self_check import validate_attention_calibration_result,validate_hif4_params


def main():
    torch.set_num_threads(4)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    reference=load_solution(ROOT/'solution/solution_v324.py');install(reference)
    roots={gi:reference.hif4_calibration_attention(bank[gi]['calib'],*[bank[gi][k] for k in ('q_num_heads','kv_num_heads','head_dim')]) for gi in (14,49)}
    rows=[]
    for version,mode in MODES.items():
        m=load_solution(ROOT/f'solution/solution_v{version}.py')
        other=load_solution(ROOT/f'solution/solution_v{version}.py')
        gi=14 if version==326 else 49;g=bank[gi]
        qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        root=roots[gi];before=copy.deepcopy(root)
        if mode in ('q_sharp','q_flat','both_flat','both_shrink'):
            proposed=metric_states(reference,root,mode,qh,kh,d)
        else:
            family='nonorthogonal' if mode.startswith(('balance_','whiten_')) else 'orthogonal'
            r=transforms(reference,g['calib'],root,qh,kh,d,family)[mode]
            if family=='orthogonal':proposed=reference._lr_states(root,r,qh,kh,d)
            else:proposed,_=nonorthogonal_states(reference,root,r,qh,kh,d)
        expected=reference._lr_guard(g['calib'][-1],root,proposed,qh,kh,d)
        start=time.perf_counter();actual=m._rg_apply(g['calib'],root,qh,kh,d)
        guard_seconds=time.perf_counter()-start
        assert not m.RG_DEBUG.get('error')
        assert tree_equal(expected,actual),(version,'research mismatch')
        assert tree_equal(before,root) and tree_equal(root['v_state'],actual['v_state'])
        assert not validate_attention_calibration_result(actual,'probe')
        returns=0
        for s in g['test']:
            for role in 'qkv':
                args=(*s[role],qh if role=='q' else kh,d,actual[role+'_state'])
                out=getattr(m,'hif4_dynamic_quantize_'+role)(*args)
                compare=getattr(other,'hif4_dynamic_quantize_'+role)(*args)
                assert tree_equal(out,compare)
                assert not validate_hif4_params(out,s[role][0].shape,role)
                returns+=1
        row=dict(version=version,mode=mode,group=gi,research_exact=True,independent_exact=True,
                 returns=returns,guard_seconds=guard_seconds,debug=copy.deepcopy(m.RG_DEBUG))
        rows.append(row);print(version,'PASS',row['debug'].get('adopted'),flush=True)
    (ROOT/'docs/v326-v338-production-probes.json').write_text(json.dumps(dict(complete=True,rows=rows,
        warning='Bounded actual-source equivalence and independent-module validation, not full 300-case regression. Guard timing includes diagnostics, not online runtime prediction.'),indent=2)+'\n')


if __name__=='__main__':main()
