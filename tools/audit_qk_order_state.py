"""Inspect state deltas before attempting safe Q/K ablations."""
import json
import torch
from probe_v319_value import ROOT,load_solution
from probe_linear_deployed_joint import tree_equal


def difference(a,b,path=''):
    if tree_equal(a,b):return []
    if isinstance(a,dict) and isinstance(b,dict):
        out=[]
        for key in sorted(set(a)|set(b)):
            if key not in a or key not in b:out.append({'path':path+'/'+key,'presence':[key in a,key in b]})
            else:out.extend(difference(a[key],b[key],path+'/'+key))
        return out
    if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):
        info={'path':path,'shapes':[list(a.shape),list(b.shape)],'dtypes':[str(a.dtype),str(b.dtype)]}
        if a.shape==b.shape:
            info['max_abs_delta']=float((a.float()-b.float()).abs().max()) if a.numel() else 0.
        if a.numel()<=16 and b.numel()<=16:info['values']=[a.tolist(),b.tolist()]
        return [info]
    return [{'path':path,'values':[a,b]}]


def main():
    torch.set_num_threads(4)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    m=load_solution(ROOT/'solution/solution_v324.py');rows=[]
    for gi,order in ((49,[0,1,2,4,3]),(14,[4,3,2,1,0]),(19,[4,3,2,1,0])):
        g=bank[gi];dims=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        a=m.hif4_calibration_attention(g['calib'],*dims)
        b=m.hif4_calibration_attention([g['calib'][i] for i in order],*dims)
        row={'group':gi,'order':order,'differences':difference(a,b)}
        rows.append(row);print(json.dumps(row),flush=True)
    (ROOT/'docs/qk-order-state-diff.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
