"""Sequence-level adoption diagnostic; fixed candidate, no test-dependent selection."""
import argparse,contextlib,copy,io,json
import torch
from probe_v319_value import ROOT,load_solution
from probe_linear_deployed_joint import tree_equal
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result


def main():
    p=argparse.ArgumentParser();p.add_argument('--groups',default='0,1,5,14,19,49')
    p.add_argument('--output',default='docs/calibration-coverage-screen.json');a=p.parse_args()
    torch.set_num_threads(4)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    m=load_solution(ROOT/'solution/solution_v343.py');rows=[]
    for gi in map(int,a.groups.split(',')):
        g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        parent=m._ms_parent_calibration(g['calib'],qh,kh,d);saved=copy.deepcopy(parent)
        records=[];ratios=[];votes=[]
        for index in range(max(2,len(g['calib'])-3),len(g['calib'])):
            st=m._ms_guard(g['calib'][:index+1],parent,qh,kh,d)
            debug=copy.deepcopy(m.MS_DEBUG)
            assert not debug['error'] and tree_equal(saved,parent)
            if 'hold_ratio' not in debug:continue
            ratio=torch.tensor(debug['hold_ratio'])
            active=st['v_state'].get('mean_search',{}).get('active',torch.zeros(kh,dtype=torch.bool)) if isinstance(st['v_state'],dict) else torch.zeros(kh,dtype=torch.bool)
            records.append(dict(calibration_index=index,length=g['calib'][index]['v'][0].shape[0],ratio=ratio.tolist(),active=active.tolist()))
            ratios.append(ratio);votes.append(active)
        if not records:
            rows.append(dict(group=gi,skipped=True));continue
        ratio=torch.stack(ratios);vote=torch.stack(votes);finite=torch.isfinite(ratio).all(0)
        policies=dict(last=vote[-1],consensus=(vote.sum(0)>=min(2,len(votes)))&finite,
            average=(ratio.mean(0)<.995)&(ratio.max(0).values<1.05)&finite,
            bounded_union=vote.any(0)&(ratio.max(0).values<1.25)&finite)
        baseline=load_solution(ROOT/'solution/solution_v324.py')
        baseline.hif4_calibration_attention=lambda *_,_st=parent:copy.deepcopy(_st)
        with contextlib.redirect_stdout(io.StringIO()):old,_=evaluate_attention(baseline,[g])
        row=dict(group=gi,calibration=records,baseline=old,policies={})
        for policy,active in policies.items():
            st=parent
            if bool(active.any()):
                st=dict(parent,v_state=dict(mean_search=dict(active=active,penalty=.01,pairs=0,parent=parent['v_state'])))
            assert not validate_attention_calibration_result(st,'coverage')
            m.hif4_calibration_attention=lambda *_,_st=st:copy.deepcopy(_st)
            with contextlib.redirect_stdout(io.StringIO()):scores,_=evaluate_attention(m,[g])
            row['policies'][policy]=dict(active=active.tolist(),scores=scores,delta=sum(scores)-sum(old))
        rows.append(row);print(gi,records,{k:(v['active'],v['delta']) for k,v in row['policies'].items()},flush=True)
    report=dict(complete=True,rows=rows,totals={k:sum(r.get('policies',{}).get(k,{}).get('delta',0) for r in rows) for k in ('last','consensus','average','bounded_union')},
        warning='Bounded local diagnostic with modeled activations, not online coverage telemetry or a deployable timing estimate. Parent calibration unchanged; sequence-level gate evaluation is not whole-pipeline independent validation.')
    (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n');print(report['totals'],flush=True)


if __name__=='__main__':main()
