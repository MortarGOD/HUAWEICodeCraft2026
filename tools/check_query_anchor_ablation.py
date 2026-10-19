"""Matched bank and search budget: cross-operand versus anchored objectives."""
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from self_check import validate_hif4_params

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    report={'complete':False,'rows':[],'hashes':{str(v):hashlib.sha256((ROOT/f'solution/solution_v{v}.py').read_bytes()).hexdigest() for v in (387,396,398)}}
    for name,path,groups in [('structural','local_fitted_models_attention_feedback_v2',[10,12]),('stress','local_fitted_archive/attention_feedback_v2_before_structural_20260906',[4])]:
        bank=torch.load(ROOT/path/'attn.pt',weights_only=False)
        for gi in groups:
            g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
            old=load_solution(ROOT/'solution/solution_v396.py');new=load_solution(ROOT/'solution/solution_v398.py')
            parent=old._wr_parent_calibration(g['calib'],qh,kh,d)
            frozen=old._wr_fit(g['calib'],parent,qh,kh,d)
            lean={k:frozen[k] for k in ('key','value','logmass','active')}
            hold=g['calib'][-1];raw=[old.dequantize_nvfp4(*hold[r]).float() for r in 'qkv']
            x=old._kt_input(raw[0],parent['q_state'],qh,d,'q')
            base=old._wr_parent_q(*hold['q'],qh,d,parent['q_state'])
            answers={'parent':base,'cross':old._wr_refine(raw[0],x,base,frozen,qh,d),'anchor':new._qa_refine(raw[0],x,base,lean,qh,d)}
            for label,p in answers.items():assert not validate_hif4_params(p,hold['q'][0].shape,label)
            decoded={label:old._deq_params(p).float() for label,p in answers.items()}
            changed={label:(q!=decoded['parent']).any(-1) for label,q in decoded.items()}
            ix=torch.where(changed['cross']|changed['anchor'])[0]
            row=dict(bank=name,group=gi,query_rows=int(ix.numel()),changed_queries={k:int(v.sum()) for k,v in changed.items()},methods={})
            if ix.numel()>1:
                target=old._joint_output(raw[0][ix],raw[1],raw[2],qh,kh,d,ix,query_subset=True)
                key=old._deq_params(old.hif4_dynamic_quantize_k(*hold['k'],kh,d,parent['k_state']))
                value=old._deq_params(old.hif4_dynamic_quantize_v(*hold['v'],kh,d,parent['v_state']))
                costs={}
                for label,q in decoded.items():
                    e=(old._joint_output(q[ix],key,value,qh,kh,d,ix,query_subset=True)-target).reshape(ix.numel(),qh,d).square().mean(-1)
                    costs[label]=e
                for label,e in costs.items():
                    fold=torch.stack((e[::2].mean(0),e[1::2].mean(0)))
                    before=torch.stack((costs['parent'][::2].mean(0),costs['parent'][1::2].mean(0)))
                    row['methods'][label]=dict(union_query_loss_ratio=float(e.mean()/costs['parent'].mean()),twofold_head_passes=int((fold<before*.995).all(0).sum()))
            report['rows'].append(row);print(row,flush=True)
            (ROOT/'docs/v398-objective-ablation.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True
    (ROOT/'docs/v398-objective-ablation.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
