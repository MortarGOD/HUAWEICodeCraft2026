"""Check sampled-query training against the actual full dynamic-Q path."""
import json
from pathlib import Path
import torch
from probe_learned_rotation import configured


def main():
    root=Path(__file__).resolve().parents[1]
    m=configured(.03,4,64,'output',64,0)
    bank=torch.load(root/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    g=bank[5];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
    states=m._lr_parent_calibration(g['calib'],qh,kh,d)
    rotation=m._lr_train(g['calib'],states,qh,kh,d)
    proposed=m._lr_states(states,rotation,qh,kh,d)
    records=[]
    for tag,sample in [('fit',g['calib'][-2]),('hold',g['calib'][-1]),('test_short',g['test'][0]),('test_long',g['test'][-1])]:
        rows=sample['q'][0].shape[0];ix=torch.linspace(0,rows-1,min(64,rows)).long()
        full=m._deq_params(m.hif4_dynamic_quantize_q(*sample['q'],qh,d,proposed['q_state']))[ix]
        subset=m._joint_quant_queries(sample,proposed['q_state'],qh,kh,d,ix)
        rawq=m.dequantize_nvfp4(*sample['q']).float()[ix]
        rawk=m.dequantize_nvfp4(*sample['k']).float()
        oldq=m._cov_transform(rawq,states['q_state'],qh,kh,d,'q')
        oldk=m._cov_transform(rawk,states['k_state'],kh,kh,d,'k').repeat_interleave(qh//kh,0)
        newq=m._cov_transform(rawq,proposed['q_state'],qh,kh,d,'q')
        newk=m._cov_transform(rawk,proposed['k_state'],kh,kh,d,'k').repeat_interleave(qh//kh,0)
        before=oldq@oldk.transpose(1,2);after=newq@newk.transpose(1,2)
        records.append(dict(sample=tag,rows=rows,full_numpy_branch=qh*rows<=4096,
                            subset_numpy_branch=qh*ix.numel()<=4096,
                            different_quantized_values=int((full!=subset).sum()),
                            quantized_max_error=float((full-subset).abs().max()),
                            continuous_logit_relative_mse=float((before-after).square().mean()/before.square().mean().clamp_min(1e-30))))
    report=dict(group=5,records=records,warning='Selected-group numerical audit only; does not prove every shape/backend is bit-exact.')
    (root/'docs/learned-rotation-deployment-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
