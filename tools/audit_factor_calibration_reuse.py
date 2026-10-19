"""Read-only K reuse opportunity audit. Never bypasses a quantizer call."""
import copy,json,time
import torch
from probe_v319_value import ROOT,load_solution
from probe_linear_deployed_joint import tree_equal


def audit(group):
    m=load_solution(ROOT/'solution/solution_v350.py')
    original=m.hif4_dynamic_quantize_k;records=[];events=[];phase='parent'
    def traced(k_quant,k_scale,kv_num_heads,head_dim,k_state):
        inputs=[r for r in records if r['quant'] is k_quant and r['scale'] is k_scale
                and r['dims']==(kv_num_heads,head_dim)]
        matches=[r for r in inputs if tree_equal(r['state'],k_state)]
        snapshot=copy.deepcopy(k_state)
        start=time.perf_counter()
        out=original(k_quant,k_scale,kv_num_heads,head_dim,k_state)
        elapsed=time.perf_counter()-start
        assert tree_equal(snapshot,k_state)
        equal=all(tree_equal(r['out'],out) for r in matches)
        assert equal
        events.append(dict(phase=phase,tokens=k_quant.shape[0],input_matches=len(inputs),
            state_matches=len(matches),parent_matches=sum(r['phase']=='parent' for r in matches),
            equal=equal,compute_seconds=elapsed))
        if len(records)<64:
            records.append(dict(quant=k_quant,scale=k_scale,dims=(kv_num_heads,head_dim),
                                state=snapshot,out=copy.deepcopy(out),phase=phase))
        return out
    m.hif4_dynamic_quantize_k=traced
    dims=[group[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
    try:
        parent=m._am_parent_calibration(group['calib'],*dims)
        phase='factor'
        state=m._am_guard(group['calib'],parent,*dims)
    finally:
        m.hif4_dynamic_quantize_k=original
    reference=load_solution(ROOT/'solution/solution_v350.py')
    expected=reference.hif4_calibration_attention(group['calib'],*dims)
    assert tree_equal(expected,state)
    factor=[e for e in events if e['phase']=='factor']
    return dict(events=events,record_cap=64,cap_reached=len(records)>=64,
                factor_calls=len(factor),factor_parent_hits=sum(e['parent_matches']>0 for e in factor),
                factor_any_hits=sum(e['state_matches']>0 for e in factor),state_equal=True)


def main():
    torch.set_num_threads(4);rows=[]
    for dataset,groups in [('local_fitted_models_attention_feedback_v2',(0,1,14)),
                           ('local_fitted_models_covariance_v1',(0,24,25))]:
        bank=torch.load(ROOT/dataset/'attn.pt',weights_only=False)
        for gi in groups:
            row=dict(dataset=dataset,group=gi,**audit(bank[gi]));rows.append(row)
            print(dataset,gi,{k:v for k,v in row.items() if k!='events'},flush=True)
    (ROOT/'docs/factor-calibration-reuse-audit.json').write_text(json.dumps(dict(complete=True,rows=rows,
        warning='Public K entry instrumentation only; no claims about private aliases or all cases. All calls execute normally; snapshots/copies are audit overhead, not a cache benchmark.'),indent=2)+'\n')


if __name__=='__main__':main()
