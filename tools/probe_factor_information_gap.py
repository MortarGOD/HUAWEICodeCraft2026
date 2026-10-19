"""Offline information ablations; oracle modes are not independent-interface candidates."""
import argparse,json
import torch
from probe_v319_value import ROOT,load_solution


def prototypes(m,samples,parent,qh,kh,d):
    pool=[s for s in samples[:-1] if 4<=s['v'][0].shape[0]<=2048
          and s['v'][0].numel()<=4194304][-2:]
    qs=[]
    for s in pool:
        t=s['v'][0].shape[0];ix=torch.linspace(0,t-1,min(16,t)).long()
        q=m._joint_quant_queries(s,parent['q_state'],qh,kh,d,ix)
        qs.append(q.reshape(-1,kh,qh//kh,d).permute(1,0,2,3).reshape(kh,-1,d))
    q=torch.cat(qs,1)
    return q[:,torch.linspace(0,q.shape[1]-1,min(m._AF_RANK,q.shape[1])).long()]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--groups',default='0,1,5,14,19,49');args=parser.parse_args()
    torch.set_num_threads(4)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    m=load_solution(ROOT/'solution/solution_v350.py');rows=[]
    for gi in map(int,args.groups.split(',')):
        g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        parent=m._am_parent_calibration(g['calib'],qh,kh,d)
        model=m._am_fit(g['calib'],parent,qh,kh,d)
        proto=prototypes(m,g['calib'],parent,qh,kh,d)
        for split,s in [('calibration_hold',g['calib'][-1]),('test_last',g['test'][-1])]:
            t=s['v'][0].shape[0];ix=torch.linspace(0,t-1,min(256,t)).long()
            q=m._joint_quant_queries(s,parent['q_state'],qh,kh,d,ix)
            k=m._deq_params(m.hif4_dynamic_quantize_k(*s['k'],kh,d,parent['k_state']))
            rawq=m.dequantize_nvfp4(s['q'][0][ix],s['q'][1][ix]).float()
            rawk=m.dequantize_nvfp4(*s['k']).float()
            rawv=m.dequantize_nvfp4(*s['v']).float()
            v=rawv.reshape(t,kh,d).transpose(0,1)
            actual=m._vr_probs(q,k,qh,kh,d).reshape(kh,-1,t)
            truth=m._vr_probs(rawq,rawk,qh,kh,d).reshape(kh,-1,t)
            params=m._am_parent_v(*s['v'],kh,d,parent['v_state'])
            base=m._deq_params(params).reshape(t,kh,d).transpose(0,1)
            step=(m._params_unit_flat(params)*.25).reshape(t,kh,d).transpose(0,1)
            target=truth@v;initial=(actual@base-target).square().sum()
            logits=proto@k.reshape(t,kh,d).permute(1,2,0)/(d**.5)
            kernels=dict(student=m._af_probability(rawv,model,kh,d),
                prototype_exact_clipped=logits.clamp(-40,40).softmax(-1),
                prototype_exact=logits.softmax(-1),actual_queries=actual[:,::2])
            mean=actual.mean(1);top=mean.topk(8,dim=1)
            norms=v.square().mean(-1).sqrt()
            row=dict(group=gi,split=split,tokens=t,modes={},
                actual_top_positions=top.indices.tolist(),actual_top_mass=top.values.tolist(),
                first8_mass=mean[:,:8].sum(1).tolist(),last8_mass=mean[:,-8:].sum(1).tolist(),
                top_value_norm_positions=norms.topk(8,dim=1).indices.tolist())
            for name,p in kernels.items():
                out=m._af_search(v,base,step,p,1.,8)
                error0=base-v;error1=out-v
                before=(p@error0).square().sum()/p.shape[1]+error0.square().sum()/t**2
                after=(p@error1).square().sum()/p.shape[1]+error1.square().sum()/t**2
                assert after<=before+1e-5*before.clamp_min(1e-8)
                assert torch.isfinite(out).all()
                row['modes'][name]=dict(final_output_ratio=float((actual@out-target).square().sum()/initial),
                    heldout_query_ratio=float((actual[:,1::2]@out-target[:,1::2]).square().sum()/
                                              (actual[:,1::2]@base-target[:,1::2]).square().sum()),
                    mean_probability_l1=float((p.mean(1)-actual.mean(1)).abs().sum(1).mean()),
                    normalized_concentration=float(t*p.square().sum(-1).mean()),
                    search_objective_ratio=float(after/before))
            rows.append(row);print(json.dumps(row),flush=True)
    suffix='' if args.groups=='0,1,5,14,19,49' else '-focused'
    (ROOT/f'docs/factor-information-gap{suffix}.json').write_text(json.dumps(dict(complete=True,rows=rows,
        warning='Offline oracle modes use current Q/K. Only student is deployable. Six groups, one hold and one test each; modeled activations, no online score inference.'),indent=2)+'\n')


if __name__=='__main__':main()
