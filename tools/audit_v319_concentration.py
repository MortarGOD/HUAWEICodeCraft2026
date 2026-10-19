"""Calibration-only routing diagnostic, never uses test labels for features."""
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution


def main():
    torch.set_num_threads(4)
    m=load_solution(ROOT/'solution/solution_v318.py')
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    records=[]
    for gi,g in enumerate(bank):
        qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        s=g['calib'][-1];rows=s['q'][0].shape[0]
        ix=torch.linspace(0,rows-1,min(rows,16)).long()
        q=m.dequantize_nvfp4(s['q'][0][ix],s['q'][1][ix]).float().reshape(-1,qh,d).transpose(0,1)
        k=m.dequantize_nvfp4(*s['k']).float().reshape(rows,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
        prob=torch.softmax(q@k.transpose(1,2)/(d**.5),-1)
        factor=rows*prob.square().sum(-1).mean(-1).reshape(kh,qh//kh).mean(-1)
        item=dict(group=gi,concentration=factor.tolist(),eligible=(factor<=16).tolist())
        records.append(item);print(item,flush=True)
    (ROOT/'docs/v319-concentration-audit.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
