"""Post-push A3 prediction/output transmission diagnostic; real peers offline only."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from hif4_quant import dequantize_nvfp4

def main():
    torch.set_num_threads(4);rows=[]
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    for gi in (0,10,19,24,49):
        g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        m=load(430,'concentration');trials=[];guard=m._ac_guard
        def capture(hold,parent,state,*a):
            trials.append(state);return guard(hold,parent,state,*a)
        m._ac_guard=capture
        state=m.hif4_calibration_attention(g['calib'],qh,kh,d)
        if not trials:
            rows.append({'group':gi,'fit_triggered':False});continue
        st=trials[-1]
        for ti,s in enumerate(g['test']):
            t=s['v'][0].shape[0];nq=min(m._VK_QUERIES,t);ix=torch.linspace(0,t-1,nq).long()
            q=dequantize_nvfp4(s['q'][0][ix],s['q'][1][ix]).float().reshape(nq,kh,qh//kh,d).permute(1,2,0,3)
            k=dequantize_nvfp4(*s['k']).float().reshape(t,kh,d).transpose(0,1)
            truth=torch.softmax(q@k[:,None].transpose(-1,-2)/d**.5,-1)
            x=dequantize_nvfp4(*s['v']).float();metrics=[]
            for p in (m._ac_parent_probability(x,st,kh),m._vk_probabilities(x,st,kh)):
                pp=p.reshape(kh,qh//kh,nq,-1,t)
                concentration=pp.square().sum(-1).mean((1,2,3))
                reference=truth.square().sum(-1).mean((1,2))
                mean=pp.mean(3)
                kl=(truth*(truth.clamp_min(1e-30).log()-mean.clamp_min(1e-30).log())).sum(-1).mean((1,2))
                top=(truth.argmax(-1)==mean.argmax(-1)).float().mean((1,2))
                metrics.append({'concentration_log_error':(concentration/reference).log().abs().tolist(),'KL':kl.tolist(),'top1':top.tolist()})
            rows.append({'group':gi,'test':ti,'tokens':t,'temperature':st['a430_temperature'].tolist(),'proposal_active':st['a430_active'].tolist(),
                         'final_selected':isinstance(state['v_state'],dict) and 'a430_temperature' in state['v_state'],'metrics':metrics})
    out={'rows':rows,'warning':'Selected systematic/known diagnostic groups; parent vs calibrated proposal, not an online forecast. Real test Q/K are used only by this diagnostic.'}
    (ROOT/'docs/attention-concentration-diagnostic.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
