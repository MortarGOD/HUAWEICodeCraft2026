"""Exact sampled-query QK/Value error decomposition using public quantizers."""
import argparse,json,hashlib,sys
import torch
from probe_v319_value import ROOT,load_solution
sys.path.insert(0,str(ROOT))
from tools.verify_attention_campaign import role_calibration


def main():
    p=argparse.ArgumentParser();p.add_argument('--groups',default='0,1,5,14,19,49')
    p.add_argument('--version',type=int,default=324)
    p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--isolated',action='store_true')
    p.add_argument('--decoy',default='block_resample',choices=['row_permute','block_resample'])
    p.add_argument('--output',default='docs/attention-error-budget.json');a=p.parse_args()
    torch.set_num_threads(4)
    bank_path=ROOT/a.dataset/'attn.pt'
    bank=torch.load(bank_path,weights_only=True)
    source=ROOT/f'solution/solution_v{a.version}.py'
    m=load_solution(source);rows=[]
    for gi in map(int,a.groups.split(',')):
        group=bank[gi];qh,kh,d=[group[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        state=role_calibration(m,group,a.isolated,a.decoy);cases=[]
        for ci,sample in enumerate(group['test']):
            t=sample['q'][0].shape[0];ix=torch.linspace(0,t-1,min(32,t)).long()
            rawq=m.dequantize_nvfp4(sample['q'][0][ix],sample['q'][1][ix]).float()
            rawk=m.dequantize_nvfp4(*sample['k']).float();rawv=m.dequantize_nvfp4(*sample['v']).float()
            quant={role:m._deq_params(getattr(m,'hif4_dynamic_quantize_'+role)(*sample[role],qh if role=='q' else kh,d,state[role+'_state'])) for role in 'qkv'}
            prob=m._vr_probs(rawq,rawk,qh,kh,d).double()
            pq=m._vr_probs(quant['q'][ix],quant['k'],qh,kh,d).double()
            def value(v):return v.reshape(-1,kh,d).transpose(0,1).repeat_interleave(qh//kh,0).double()
            v=value(rawv);vq=value(quant['v']);target=prob@v
            joint=pq@vq-target;qk=(pq-prob)@v;ve=pq@(vq-v)
            residual=joint-qk-ve
            scale=joint.abs().amax().clamp_min(1e-15)
            identity_error=float(residual.abs().amax()/scale)
            assert identity_error<1e-8,identity_error
            energy=target.square().sum().clamp_min(1e-30)
            def normalized(x):return float(x.square().sum()/energy)
            total=normalized(joint);qkloss=normalized(qk);vloss=normalized(ve)
            cross=float(2*(qk*ve).sum()/energy)
            assert abs(total-qkloss-vloss-cross)<max(1e-12,total*1e-9)
            cases.append(dict(case=ci,query_length=t,key_length=rawk.shape[0],sampled_queries=ix.numel(),
                total=total,qk=qkloss,carried_value=vloss,cross=cross,
                ideal_value=qkloss,ideal_qk=normalized(prob@(vq-v)),
                concentration=float(pq.square().sum(-1).mean()),identity_relative_error=identity_error))
        means={k:sum(c[k] for c in cases)/len(cases) for k in ('total','qk','carried_value','cross','ideal_qk')}
        rows.append(dict(group=gi,cases=cases,mean_normalized_mse=means))
        print(gi,means,flush=True)
    (ROOT/a.output).write_text(json.dumps(dict(complete=True,rows=rows,version=a.version,
        dataset=a.dataset,dataset_sha256=hashlib.sha256(bank_path.read_bytes()).hexdigest(),
        isolated=a.isolated,decoy=a.decoy,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        warning='Bounded cases and up to 32 sampled queries per case, full keys, actual public quantizers. Dataset provenance must be checked separately; not online decomposition. Signed cross term means component ratios are not independent score contributions. Ideal-value/ideal-QK ablations are not attainable quantized upper bounds.'),indent=2)+'\n')


if __name__=='__main__':main()
