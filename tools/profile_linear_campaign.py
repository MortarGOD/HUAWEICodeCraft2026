"""L5 bounded helper costs plus L3 natural accepted-state mechanism attribution."""
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.test_v424 import packed_pair

def main():
    torch.set_num_threads(4)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/linear.pt',weights_only=True)
    m=load(428,'profile');group=bank[4];trials=[]
    fn=m._lr_rotation_trial
    def capture(parent,r):
        out=fn(parent,r);trials.append((parent,out));return out
    m._lr_rotation_trial=capture
    result=m.hif4_calibration_and_quantize_weight(*group['weight'],group['calib_activation_list'])
    rows=[]
    for parent,trial in trials:
        rows.append({'changed_W':int((m._linear_score_decode(parent['weight_params'])!=m._linear_score_decode(trial['weight_params'])).sum()),
                     'changed_P':int((parent['activation_state']['unrestricted_continuous_projection']!=trial['activation_state']['unrestricted_continuous_projection']).sum()),
                     'selected_final':trial is result})
    out={'natural_group_4_rotation_trials':rows,'large_helpers':[],'timing_note':'4 threads local isolated helper wall time, not full API or Kunpeng estimate.'}
    for width in (1024,2048):
        g=torch.Generator().manual_seed(510+width);j=load(427,f'large{width}')
        b=torch.randn((2*width,width),generator=g)*.1
        params=j._quantize_linear_values(b,offsets=j._LINEAR_WEIGHT_SCALE_OFFSETS)
        teacher=j._linear_score_decode(params).float()
        state={'unrestricted_target_weight':teacher.bfloat16(),'unrestricted_continuous_projection':torch.eye(width)}
        t=time.perf_counter();compiled=j._lr_compile(state,params,state['unrestricted_continuous_projection']);compile_s=time.perf_counter()-t
        dec=[]
        for _ in range(4):
            z=torch.randn((16,width),generator=g).bfloat16();target=z.float()@teacher.T
            dec.append((z.float(),z,target,target.square().sum()))
        result={'weight_params':params,'activation_state':compiled}
        t=time.perf_counter();proposal=j._lr_joint_proposal(result,dec);proposal_s=time.perf_counter()-t
        deployed=j._linear_score_decode(proposal).float();gram=deployed.T@deployed;gram.diagonal().add_(.001*gram.diagonal().mean())
        t=time.perf_counter();p=torch.linalg.solve(gram,(teacher.T@deployed).T).T;solve_s=time.perf_counter()-t
        out['large_helpers'].append({'input_width':width,'output_width':2*width,'compile_seconds':compile_s,'joint_proposal_seconds':proposal_s,'projection_solve_seconds':solve_s,'finite':bool(torch.isfinite(p).all())})
    (ROOT/'docs/linear-campaign-profile.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
