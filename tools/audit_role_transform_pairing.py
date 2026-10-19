"""Top-level continuous Q/K transform compatibility; not a score attribution proof."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from tools.check_role_decoy_calibration import role_samples


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    rows=[]
    for bank,path,gi in [('structural','local_fitted_models_attention_feedback_v2',10),
                         ('stress','local_fitted_archive/attention_feedback_v2_before_structural_20260906',49)]:
        g=torch.load(ROOT/path/'attn.pt',weights_only=False)[gi]
        qh,kh,d=(g[k] for k in ('q_num_heads','kv_num_heads','head_dim'))
        for mode in ('paired','row_permute','block_resample'):
            m=load_solution(ROOT/'solution/solution_v400.py')
            if mode=='paired':states=m.hif4_calibration_attention(g['calib'],qh,kh,d)
            else:
                states={}
                for ri,role in enumerate('qk'):
                    m=load_solution(ROOT/'solution/solution_v400.py')
                    result=m.hif4_calibration_attention(role_samples(g['calib'],role,mode,20260906+ri),qh,kh,d)
                    states[role+'_state']=result[role+'_state']
            matrices=[]
            for role,nh in (('q',qh),('k',kh)):
                st=dict(states[role+'_state']);st.pop('temperature_gain',None)
                mat=m._kt_input(torch.eye(d).repeat(1,nh),st,nh,d,role).reshape(d,nh,d).transpose(0,1)
                matrices.append(mat[::qh//kh] if role=='q' else mat)
            product=matrices[0]@matrices[1].transpose(1,2)
            error=(product-torch.eye(d)).square().sum((1,2)).sqrt()/d**.5
            row=dict(bank=bank,group=gi,mode=mode,relative_identity_error=error.tolist(),
                note='Top-level pre-translation linear maps, temperature removed. Does not include quantization, routing or prove causality of score loss.')
            rows.append(row);print(row,flush=True)
    (ROOT/'docs/role-transform-pairing-audit.json').write_text(json.dumps(dict(rows=rows,decoy_source='unknown; sensitivity hypotheses only'),indent=2)+'\n')


if __name__=='__main__':main()
