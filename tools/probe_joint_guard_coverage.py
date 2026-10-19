"""Offline full-query upper-budget diagnostic for the existing joint guard."""
import contextlib,copy,io,json,types,time
import torch
from probe_v319_value import ROOT,load_solution
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result


def main():
    torch.set_num_threads(4)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    path=ROOT/'solution/solution_v324.py';source=path.read_text()
    needle='ix = torch.linspace(0, rows - 1, min(rows, 64)).long()\n        target = _joint_output(*raw, qh, kh, d, ix)'
    assert source.count(needle)==1
    changed=source.replace(needle,'ix = torch.arange(rows)\n        target = _joint_output(*raw, qh, kh, d, ix)')
    base=load_solution(path);full=types.ModuleType('joint_full_query_research')
    exec(compile(changed,'joint_full_query_research','exec'),full.__dict__)
    calibrators=[base.hif4_calibration_attention,full.hif4_calibration_attention];rows=[]
    for gi in (0,1,5,14,19,49):
        g=bank[gi];dims=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')];variants={}
        for name,m,cal in zip(('base','full'),(base,full),calibrators):
            start=time.perf_counter();state=cal(g['calib'],*dims);elapsed=time.perf_counter()-start
            assert not validate_attention_calibration_result(state,'joint_coverage')
            debug=copy.deepcopy(m.JOINT_DEBUG)
            m.hif4_calibration_attention=lambda *_,_st=state:copy.deepcopy(_st)
            with contextlib.redirect_stdout(io.StringIO()):scores,_=evaluate_attention(m,[g])
            variants[name]={'scores':scores,'joint_debug':debug,'calibration_seconds':elapsed}
        row={'group':gi,'variants':variants,'delta':sum(variants['full']['scores'])-sum(variants['base']['scores'])}
        rows.append(row);print(row,flush=True)
    report={'complete':True,'rows':rows,'delta':sum(r['delta'] for r in rows),
            'warning':'Full-query diagnostic only; no deployment timing or online prediction. Modeled activations.'}
    (ROOT/'docs/joint-guard-full-query.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
