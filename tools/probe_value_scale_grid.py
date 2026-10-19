"""Broader native scale grids, independently guarded by final Attention loss."""
import argparse,contextlib,copy,io,json
import torch
from probe_v319_value import ROOT,load_solution
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result,validate_hif4_params


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preserve-parent',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    m=load_solution(ROOT/'solution/solution_v343.py');rows=[]
    for gi in (0,1,5,14,19,49):
        g=bank[gi];dims=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        parent=m._ms_parent_calibration(g['calib'],*dims)
        m.hif4_calibration_attention=lambda *_,_st=parent:copy.deepcopy(_st)
        with contextlib.redirect_stdout(io.StringIO()):old,_=evaluate_attention(m,[g])
        row=dict(group=gi,baseline=old,modes={})
        for name,offsets in (('native6',None),('wide16',range(-1,3)),('wide24',range(-2,4))):
            grid=m.CAND_GRID if offsets is None else tuple((off,sig) for off in offsets for sig in m.E6M2_SIG)
            def raw_quant(raw,_grid=grid):
                return m._quantize_weighted(raw,torch.ones(1,raw.shape[1]),grid=_grid)
            def quant(raw,base,st,kh,d,_grid=grid):
                if args.preserve_parent:
                    # Offline serial intervention only, never a submission pattern.
                    original=m._dyn_table
                    m._dyn_table=lambda x,state,has_scale:raw_quant(x)
                    try:
                        out=m._ms_parent_v(raw.to(torch.bfloat16),torch.ones(raw.shape[0],raw.shape[1]//16),kh,d,parent['v_state'])
                    finally:m._dyn_table=original
                    if name=='native6':assert all(torch.equal(out[k],v) for k,v in base.items())
                else:out=raw_quant(raw)
                active=st.get('active',torch.ones(kh,dtype=torch.bool)).repeat_interleave(d//64)
                return {k:torch.where(active.reshape(1,-1,*([1]*(v.ndim-2))),out[k],v) for k,v in base.items()}
            m._ms_quant=quant
            state=m._ms_guard(g['calib'],parent,*dims);debug=copy.deepcopy(m.MS_DEBUG)
            assert not debug['error'] and not validate_attention_calibration_result(state,'grid')
            m.hif4_calibration_attention=lambda *_,_st=state:copy.deepcopy(_st)
            with contextlib.redirect_stdout(io.StringIO()):scores,_=evaluate_attention(m,[g])
            mse=[]
            for sample in g['test']:
                raw=m.dequantize_nvfp4(*sample['v']).float()
                base=m._dyn_table(raw,None,False)
                wide=raw_quant(raw)
                assert not validate_hif4_params(wide,raw.shape,'v')
                a=(m._deq_params(base)-raw).square().sum();b=(m._deq_params(wide)-raw).square().sum()
                assert b<=a+max(float(a)*1e-6,1e-10)
                mse.append(float(b/a.clamp_min(1e-30)))
            row['modes'][name]=dict(delta=sum(scores)-sum(old),scores=scores,debug=debug,mse_ratio=mse)
        rows.append(row);print(gi,{k:(v['delta'],v['debug']['adopted'],sum(v['mse_ratio'])/5) for k,v in row['modes'].items()},flush=True)
    report=dict(complete=True,rows=rows,totals={k:sum(r['modes'][k]['delta'] for r in rows) for k in ('native6','wide16','wide24')})
    report['preserve_parent']=args.preserve_parent
    filename='value-scale-grid-preserved.json' if args.preserve_parent else 'value-scale-grid-screen.json'
    (ROOT/'docs'/filename).write_text(json.dumps(report,indent=2)+'\n');print(report['totals'],flush=True)


if __name__=='__main__':main()
