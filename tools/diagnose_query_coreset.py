"""Representative accuracy against 64-query predicted risk, not online scoring."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
torch.set_num_threads(4)
m=load_solution(ROOT/'solution/solution_v402.py');rows=[]
exec((ROOT/'tools/query_coreset_core.py').read_text(),m.__dict__)
for name,path,indices in [('structural','local_fitted_models_attention_feedback_v2',[10,12,14,18]),('stress','local_fitted_archive/attention_feedback_v2_before_structural_20260906',[4,47])]:
    bank=torch.load(ROOT/path/'attn.pt',weights_only=False)
    for gi in indices:
        g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        state=m._vk_fit(g['calib'],qh,kh,d)
        for ti,sample in enumerate(g['test']):
            x=m.dequantize_nvfp4(*sample['v']).float();t=len(x)
            f=m._vk_features(x,state);ix=torch.linspace(0,t-1,min(64,t)).long()
            logits=(f[ix][None]@state['kernel'].float())@f.T
            modes=state.get('query_sigma_modes')
            if modes is not None and bool(modes.any()):
                delta=modes.float()@f.T*2**.5;offsets=torch.cat((delta,-delta),1)
            else:offsets=torch.zeros(qh,1,t)
            dense=torch.softmax(logits[:,:,None]+offsets[:,None],-1).reshape(kh,-1,t)
            parent=m._dyn_table(x,None,has_scale=False)
            e=(x-m._deq_params(parent)).reshape(t,kh,d).transpose(0,1)
            def risk(p):return (p@e).square().mean((1,2))
            ref=risk(dense)
            losses={}
            for label,p in [('uniform4',m._cq_parent_probabilities(x,state,kh)),('coreset4',m._vk_probabilities(x,state,kh))]:
                losses[label]=float((risk(p)-ref).abs().sum()/ref.abs().sum().clamp_min(1e-30))
            rows.append(dict(bank=name,group=gi,test=ti,t=t,relative_absolute_risk_error=losses))
report=dict(complete=True,reference='64 equally spaced predicted queries with unchanged v400 residual scenarios; same standard V error; not raw Q/K or end-to-end accuracy',rows=rows)
(ROOT/'docs/v403-representative-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
for name in ('structural','stress'):
    rr=[x['relative_absolute_risk_error'] for x in rows if x['bank']==name]
    print(name,len(rr),{k:sum(x[k] for x in rr)/len(rr) for k in ('uniform4','coreset4')},'coreset closer',sum(x['coreset4']<x['uniform4'] for x in rr))
