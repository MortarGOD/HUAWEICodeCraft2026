"""Actual final-output residual student, shared-parent screening."""
import argparse,contextlib,copy,io,json
import torch
from probe_v319_value import ROOT,load_solution
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result,validate_hif4_params


def main():
    p=argparse.ArgumentParser();p.add_argument('--groups',default='0,1,5,14,19,49')
    p.add_argument('--bank',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--output',default='docs/v340-student-screen.json');a=p.parse_args()
    torch.set_num_threads(4);bank=torch.load(ROOT/a.bank/'attn.pt',weights_only=False)
    m=load_solution(ROOT/'solution/solution_v324.py')
    exec(compile((ROOT/'tools/attention_residual_student_core.py').read_text(),'student','exec'),m.__dict__)
    rows=[]
    for gi in map(int,a.groups.split(',')):
        g=bank[gi];dims=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        parent=m._rs_parent_calibration(g['calib'],*dims)
        m.hif4_calibration_attention=lambda *_,_st=parent:copy.deepcopy(_st)
        with contextlib.redirect_stdout(io.StringIO()):old,_=evaluate_attention(m,[g])
        st=m._rs_guard(g['calib'],parent,*dims)
        assert not validate_attention_calibration_result(st,'student')
        m.hif4_calibration_attention=lambda *_,_st=st:copy.deepcopy(_st)
        with contextlib.redirect_stdout(io.StringIO()):score,_=evaluate_attention(m,[g])
        for s in g['test']:
            out=m.hif4_dynamic_quantize_v(*s['v'],dims[1],dims[2],st['v_state'])
            assert not validate_hif4_params(out,s['v'][0].shape,'v')
        row=dict(group=gi,baseline=old,scores=score,delta=sum(score)-sum(old),debug=copy.deepcopy(m.RS_DEBUG))
        rows.append(row);print(gi,round(row['delta'],6),row['debug'],flush=True)
    (ROOT/a.output).write_text(json.dumps(dict(complete=True,rows=rows,total=sum(r['delta'] for r in rows)),indent=2)+'\n')


if __name__=='__main__':main()
