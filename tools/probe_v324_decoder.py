"""Actual deployed activation codes train fixed native decoder candidates."""
import argparse,copy,json,time
import torch
from probe_v319_value import ROOT,load_solution
from linear_decoder_cd_core import _linear_decoder_cd
from hif4_quant import dequantize_hif4,quantize_to_hif4
from self_check import validate_hif4_params,validate_linear_calibration_result

def main():
    p=argparse.ArgumentParser();p.add_argument('--groups',default='0,4,6,7,9');p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2');p.add_argument('--encoder-metric',action='store_true');p.add_argument('--refresh',action='store_true');p.add_argument('--output',default='docs/v324-decoder-screen.json');a=p.parse_args()
    torch.set_num_threads(4);m=load_solution(ROOT/'solution/solution_v322.py');bank=torch.load(ROOT/a.dataset/'linear.pt',weights_only=False);rows=[]
    for gi in map(int,a.groups.split(',')):
        g=bank[gi];w=m.dequantize_nvfp4(*g['weight']).float();bw=dequantize_hif4(quantize_to_hif4(w),tuple(w.shape)).float()
        parent=m.hif4_calibration_and_quantize_weight(*g['weight'],g['calib_activation_list']);states=[('parent',parent)]
        def forward(packed,state):
            pa=m.hif4_dynamic_quantize_activation(*packed,state['activation_state']);assert not validate_hif4_params(pa,packed[0].shape,'A')
            z=m._linear_score_decode(pa);b=m._linear_score_decode(state['weight_params']);x=m.dequantize_nvfp4(*packed).float();y=x@w.T
            return z.float(),y,(y-z@b.T).square().mean(-1)
        zs=[];ys=[]
        for packed in g['calib_activation_list'][:-1]:
            z,y,_=forward(packed,parent);x=m.dequantize_nvfp4(*packed).float();xb=dequantize_hif4(quantize_to_hif4(x),tuple(x.shape)).float()
            bm=(y-xb@bw.T).square().mean().clamp_min(1e-20)
            ix=torch.linspace(0,len(z)-1,min(len(z),64)).long();scale=(len(ix)*bm).rsqrt();zs.append(z[ix]*scale);ys.append(y[ix]*scale)
        z=torch.cat(zs);y=torch.cat(ys);start=time.perf_counter()
        if a.encoder_metric:
            z=parent['activation_state']['unrestricted_continuous_projection'].float()
            y=parent['activation_state']['unrestricted_target_weight'].float().T
        for ridge in (.01,.1,1.):
            for steps in (1,2):
                params=_linear_decoder_cd(m,parent['weight_params'],z,y,steps,ridge)
                state=copy.deepcopy(parent['activation_state'])
                if a.refresh:state.pop('unrestricted_continuous_projection',None)
                m._install_unrestricted_exact_target_state(state,state['unrestricted_target_weight'].float(),params)
                result=dict(weight_params=params,activation_state=state);assert not validate_linear_calibration_result(result,w.shape,'candidate')
                states.append((f'cd_r{ridge}_n{steps}',result))
        guard=[]
        for name,st in states:
            _,_,loss=forward(g['calib_activation_list'][-1],st);guard.append(dict(name=name,folds=[float(loss[j::2].mean()) for j in (0,1)]))
        ok=[i for i in range(1,len(states)) if all(guard[i]['folds'][j]<guard[0]['folds'][j]*.9999 for j in (0,1))]
        chosen=min(ok,key=lambda i:sum(guard[i]['folds'])) if ok else 0
        item=dict(group=gi,chosen=states[chosen][0],guard=guard,cases=[],research_seconds=time.perf_counter()-start)
        for ti,packed in enumerate(g['test_activation_list']):
            x=m.dequantize_nvfp4(*packed).float();y=x@w.T;xb=dequantize_hif4(quantize_to_hif4(x),tuple(x.shape)).float();bm=float((y-xb@bw.T).square().mean())
            _,_,old=forward(packed,parent);_,_,new=forward(packed,states[chosen][1]);item['cases'].append(dict(test=ti,parent=1-float(old.mean())/bm,selected=1-float(new.mean())/bm,delta=float((old.mean()-new.mean())/bm)))
        item['delta']=sum(c['delta'] for c in item['cases']);rows.append(item);print(gi,item['chosen'],item['delta'],flush=True)
    (ROOT/a.output).write_text(json.dumps(dict(complete=True,encoder_metric=a.encoder_metric,groups=rows,delta=sum(r['delta'] for r in rows),score_prediction_allowed=False),indent=2)+'\n')
if __name__=='__main__':main()
