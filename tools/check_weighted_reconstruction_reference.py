"""Fixed-group uniform/weighted/full-bank ranking and held-sequence diagnostics."""
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    m=load_solution(ROOT/'solution/solution_v396.py')
    report={'complete':False,'source_sha256':hashlib.sha256((ROOT/'solution/solution_v396.py').read_bytes()).hexdigest(),'rows':[]}
    for name,path,groups in [('structural','local_fitted_models_attention_feedback_v2',[10,12]),('stress','local_fitted_archive/attention_feedback_v2_before_structural_20260906',[4])]:
        bank=torch.load(ROOT/path/'attn.pt',weights_only=False)
        for gi in groups:
            g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
            parent=m._wr_parent_calibration(g['calib'],qh,kh,d)
            weighted=m._wr_fit(g['calib'],parent,qh,kh,d)
            hold=g['calib'][-1];rawq,rawk,rawv=[m.dequantize_nvfp4(*hold[r]).float() for r in 'qkv']
            ix=torch.linspace(0,rawq.shape[0]-1,min(32,rawq.shape[0])).long()
            base=m._wr_parent_q(*hold['q'],qh,d,parent['q_state'])
            q=m._deq_params(base)[ix].float().reshape(-1,qh,d).transpose(0,1)
            raw=rawq[ix].reshape(-1,qh,d).transpose(0,1)
            step=m._params_unit_flat(base)[ix].reshape(-1,qh,d).transpose(0,1)*.25
            full=[];uniform=[]
            for s in g['calib'][:-1][-2:]:
                kr,vr=[m.dequantize_nvfp4(*s[r]).float().reshape(-1,kh,d).transpose(0,1) for r in 'kv']
                k=m._deq_params(m.hif4_dynamic_quantize_k(*s['k'],kh,d,parent['k_state'])).float().reshape(-1,kh,d).transpose(0,1)
                v=m._deq_params(m.hif4_dynamic_quantize_v(*s['v'],kh,d,parent['v_state'])).float().reshape(-1,kh,d).transpose(0,1)
                record=dict(raw=kr/(d**.5),key=k/(d**.5),raw_value=vr,value=v,logmass=torch.zeros(kh,k.shape[1]))
                full.append(record)
                indices=torch.linspace(0,k.shape[1]-1,min(64,k.shape[1])).long()
                uniform.append({key:value[:,indices] for key,value in record.items()})
            compressed=[{key:weighted[key][:,s] for key in ('raw','key','raw_value','value','logmass')} for s in range(weighted['key'].shape[1])]
            def cost_grad(z,records):
                total=torch.zeros(z.shape[:2]);gradient=torch.zeros_like(z)
                for record in records:
                    r={key:value.repeat_interleave(qh//kh,0) for key,value in record.items()}
                    target=torch.softmax(raw@r['raw'].transpose(1,2)+r['logmass'][:,None],-1)@r['raw_value']
                    prob=torch.softmax(z@r['key'].transpose(1,2)+r['logmass'][:,None],-1)
                    residual=prob@r['value']-target
                    influence=residual@r['value'].transpose(1,2)
                    gradient+=2*(prob*(influence-(prob*influence).sum(-1,keepdim=True)))@r['key']
                    total+=residual.square().sum(-1)
                return total/len(records),gradient/len(records)
            variants=[q];labels=['parent']
            for label,records in [('uniform',uniform),('weighted',compressed),('full',full)]:
                _,grad=cost_grad(q,records)
                direction=torch.where(grad<0,1.,-1.)
                est=(grad*direction*step).masked_fill(((q/step).round()+direction).abs()>7,float('inf'))
                ids=est.topk(4,dim=-1,largest=False).indices
                for width in (1,4):
                    coord=ids[...,:width]
                    delta=(direction*step).gather(-1,coord)
                    delta=torch.where(torch.isfinite(est.gather(-1,coord)),delta,0.)
                    variants.append(q.clone().scatter_add_(-1,coord,delta));labels.append(label+str(width))
            costs={label:torch.stack([cost_grad(z,records)[0] for z in variants]) for label,records in [('uniform',uniform),('weighted',compressed),('full',full)]}
            # Each query/head independently picks among the same candidate set.
            truth=costs['full'];target=m._joint_output(rawq[ix],rawk,rawv,qh,kh,d,ix,query_subset=True)
            key=m._deq_params(m.hif4_dynamic_quantize_k(*hold['k'],kh,d,parent['k_state']))
            value=m._deq_params(m.hif4_dynamic_quantize_v(*hold['v'],kh,d,parent['v_state']))
            def held(z):
                output=m._joint_output(z.transpose(0,1).reshape(ix.numel(),qh*d),key,value,qh,kh,d,ix,query_subset=True)
                return (output-target).reshape(ix.numel(),qh,d).square().mean(-1).transpose(0,1)
            old=held(q);row=dict(bank=name,group=gi,labels=labels,methods={})
            for label,cost in costs.items():
                chosen=cost.argmin(0)
                actual=truth.gather(0,chosen[None]).squeeze(0)
                z=torch.stack(variants).gather(0,chosen[None,...,None].expand(1,qh,ix.numel(),d)).squeeze(0)
                he=held(z)
                agreement=[]
                for a in range(len(variants)):
                    for b in range(a):agreement.append(((cost[a]<cost[b])==(truth[a]<truth[b])).float().mean())
                row['methods'][label]=dict(rank_agreement=float(torch.stack(agreement).mean()),full_reference_cost=float(actual.mean()),parent_full_reference_cost=float(truth[0].mean()),held_loss_ratio=float(he.mean()/old.mean()),held_twofold_ratios=[float(he[:,j::2].mean()/old[:,j::2].mean()) for j in (0,1)])
            report['rows'].append(row);print(name,gi,row['methods'],flush=True)
            (ROOT/'docs/v396-reference-screen.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True
    (ROOT/'docs/v396-reference-screen.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
