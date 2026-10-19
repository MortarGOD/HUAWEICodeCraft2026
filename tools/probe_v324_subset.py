"""Parent-state paired subset repair screen, calibration never sees test data."""
import argparse,copy,hashlib,json,time
import torch
from build_v324_candidate import subset_candidate_source as candidate_source,ROOT
from probe_v319_value import load_solution
from hif4_quant import dequantize_hif4,quantize_to_hif4
from self_check import validate_hif4_params

def main():
    p=argparse.ArgumentParser();p.add_argument('--groups',default='0,4,6,7,9');p.add_argument('--modes',default='1,2');p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2');p.add_argument('--output',default='docs/v324-subset-screen.json');a=p.parse_args()
    torch.set_num_threads(4);parent=load_solution(ROOT/'solution/solution_v322.py');bank=torch.load(ROOT/a.dataset/'linear.pt',weights_only=False)
    mods={0:parent}
    for mode in map(int,a.modes.split(',')):
        m=load_solution(ROOT/'solution/solution_v322.py');exec(compile(candidate_source(mode),'subset','exec'),m.__dict__);mods[mode]=m
    rows=[]
    for gi in map(int,a.groups.split(',')):
        g=bank[gi];w=parent.dequantize_nvfp4(*g['weight']).float();bw=dequantize_hif4(quantize_to_hif4(w),tuple(w.shape)).float()
        calibrated=parent.hif4_calibration_and_quantize_weight(*g['weight'],g['calib_activation_list']);b=parent._linear_score_decode(calibrated['weight_params'])
        for ti,packed in enumerate(g['test_activation_list']):
            x=parent.dequantize_nvfp4(*packed).float();y=x@w.T;bx=dequantize_hif4(quantize_to_hif4(x),tuple(x.shape)).float();bm=float((y-bx@bw.T).square().mean());row=dict(group=gi,test=ti,rows=len(x),scores={},seconds={})
            for mode,m in mods.items():
                state=copy.deepcopy(calibrated['activation_state']);state['linear_subset_mode']=mode
                t=time.perf_counter();out=m.hif4_dynamic_quantize_activation(*packed,state);row['seconds'][mode]=time.perf_counter()-t
                assert not validate_hif4_params(out,x.shape,'linear');z=m._linear_score_decode(out);row['scores'][mode]=1-float((y-z@b.T).square().mean())/bm
            rows.append(row)
        print(gi,{mode:sum(r['scores'][mode]-r['scores'][0] for r in rows) for mode in mods},flush=True)
    report=dict(complete=True,dataset=a.dataset,rows=rows,deltas={mode:sum(r['scores'][mode]-r['scores'][0] for r in rows) for mode in mods},score_prediction_allowed=False,warning='Checkpoint weights and modeled activations; screen timing includes concurrent work or scoring overhead context, not final performance.')
    (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
