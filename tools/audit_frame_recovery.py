"""Audit repeated Gram recovery on a fixed real parent U; no production patch."""
import argparse
import json
from pathlib import Path
import torch
from probe_attention_gates import configured


def safe_metric(state,damp):
    if 'H' in state:
        h=state['H'].float();return (h+h.transpose(-1,-2))*.5
    u=state['u'].float()
    b=torch.linalg.solve_triangular(u,torch.eye(u.shape[-1]).expand_as(u),upper=True)
    h=b@b.transpose(-1,-2)
    ridge=h.diagonal(dim1=-2,dim2=-1).mean(-1)*(damp/(1+damp))
    return (h+h.transpose(-1,-2))*.5-ridge[:,None,None]*torch.eye(h.shape[-1])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='docs/frame-recovery-repeat-audit-checked.json')
    args_cli=parser.parse_args()
    root=Path(__file__).resolve().parents[1];m=configured('frames',0)
    g=torch.load(root/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)[47]
    args=(g['calib'],g['q_num_heads'],g['kv_num_heads'],g['head_dim'])
    parent=m._joint_parent_calibration(*args);state=parent['k_state'];assert 'H' not in state
    u=state['u'].double();b=torch.linalg.solve_triangular(u,torch.eye(u.shape[-1],dtype=torch.float64).expand_as(u),upper=True)
    h=b@b.transpose(-1,-2);ridge=h.diagonal(dim1=-2,dim2=-1).mean(-1)*(m.GPTQ_DAMP/(1+m.GPTQ_DAMP))
    reference=h-ridge[:,None,None]*torch.eye(h.shape[-1],dtype=torch.float64)
    scale=float(reference.norm());records=[];old=m._joint_raw_metric;original_u=state['u'].clone()
    for i in range(20):
        original=old(state).double();safe=safe_metric(state,m.GPTQ_DAMP).double()
        assert torch.equal(original_u,state['u'])
        records.append(dict(index=i,original_relative_error=float((original-reference).norm())/scale,
                            safe_relative_error=float((safe-reference).norm())/scale))
    trials={}
    for mode in ('original','triangular_solve'):
        m._joint_raw_metric=old if mode=='original' else lambda s:safe_metric(s,m.GPTQ_DAMP)
        trials[mode]=[]
        for _ in range(5):
            m._gate_parent_calibration(*args)
            trials[mode].append(dict(m.JOINT_DEBUG))
    report=dict(torch_version=torch.__version__,dtype=str(state['u'].dtype),shape=list(state['u'].shape),input_u_unchanged=True,
                fixed_u_trials=records,parent_repeats=trials,
                warning='Local diagnostic; safe recovery patched only this isolated module. No claim about Linux online runtime and no submission changes.')
    (root/args_cli.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
