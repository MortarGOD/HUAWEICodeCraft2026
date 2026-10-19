"""Numerical audit of carried Hessians and reciprocal pre-scales; read-only."""
import json
from pathlib import Path
import torch
from probe_gptq_covariance import configured


def main():
    root=Path(__file__).resolve().parents[1]
    bank=torch.load(root/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    m=configured('legacy');records=[]
    for gi in (1,5,49):
        g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        states=m._cov_parent_calibration(g['calib'],qh,kh,d)
        hk,hq=m._cov_fit(g['calib'],states,qh,kh,d)
        record=dict(group=gi,calibration_coverage=dict(m.COV_DEBUG),roles={})
        if 'qs' in states['q_state'] and 'ks' in states['k_state']:
            scale_product=states['q_state']['qs'].float()*states['k_state']['ks'].float()
            record['reciprocal_max_error']=float((scale_product-1).abs().max())
        for role,fresh in (('q',hk),('k',hq)):
            st=states[role+'_state']
            if 'u' not in st:continue
            u=st['u'].double()
            inv_u=torch.linalg.solve_triangular(u,torch.eye(d,dtype=torch.float64).expand_as(u),upper=True)
            damped=inv_u@inv_u.transpose(1,2)
            ridge=damped.diagonal(dim1=-2,dim2=-1).mean(-1)*(m.GPTQ_DAMP/(1+m.GPTQ_DAMP))
            implied=damped-ridge[:,None,None]*torch.eye(d,dtype=torch.float64)
            fresh=fresh.double().repeat_interleave(implied.shape[0]//fresh.shape[0],0)
            stats=dict(fresh_vs_U_implied_relative=float((fresh-implied).norm()/implied.norm().clamp_min(1e-30)))
            if 'H' in st:
                carried=st['H'].double()
                implied=implied.repeat_interleave(carried.shape[0]//implied.shape[0],0)
                average=carried.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-30)
                eigen=torch.linalg.eigvalsh((carried+carried.transpose(1,2))*.5)
                stats.update(H_dtype=str(st['H'].dtype),
                             carried_vs_U_implied_relative=float((carried-implied).norm()/implied.norm().clamp_min(1e-30)),
                             H_min_eigen_over_mean_diag=(eigen[:,0]/average).tolist(),
                             negative_eigenvalues=(eigen<0).sum(-1).tolist())
            record['roles'][role]=stats
        records.append(record)
        print(gi,record['roles'],flush=True)
    report=dict(records=records,warning='Local diagnostic, not an attribution of online score loss; indefinite BF16 H does not prove every refinement hurts.')
    (root/'docs/gptq-covariance-numeric-audit.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
