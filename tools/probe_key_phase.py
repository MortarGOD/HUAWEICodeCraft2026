"""Current-Key fixed-grid phase selection followed by unchanged GPTQ."""
import argparse,contextlib,copy,io,json
import torch
from probe_v319_value import ROOT,load_solution
from key_phase_math import phase_search
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result,validate_hif4_params
from probe_linear_deployed_joint import tree_equal


CORE='''
def _kp_quant(raw,st,kh,d,variant):
    y=_kt_input(raw,st,kh,d,'k').reshape(-1,kh,d)
    key=raw.reshape(-1,kh,d).transpose(0,1)
    prob=torch.softmax(st['kt_probe'].float()@key.transpose(1,2),-1)
    weight=(.9*prob.mean(1)+.1/raw.shape[0]).transpose(0,1)[:,:,None]
    center=(y*weight).sum(0,keepdim=True)
    for _ in range(8):
        z=y-center
        error=_deq_params(_dyn_table(z.reshape(raw.shape),None,has_scale=False)).reshape(y.shape)-z
        center=center-(error*weight).sum(0,keepdim=True)
    p=_dyn_table((y-center).reshape(raw.shape),None,has_scale=False)
    step=(_params_unit_flat(p)*.25).reshape(y.shape)
    center,_=phase_search(y,center,step,weight)
    plain={k:v for k,v in st.items() if k not in ('joint_transform','ks','rot','hc_matrix')}
    return _kt_fixed_grid((y-center).reshape(raw.shape).contiguous(),plain,kh,d,'k')
'''


def cross_guard(m,holds,parent,candidate,dims):
    """Two untouched calibration sequences; no test sample access."""
    qh,kh,d=dims;ratios=[]
    for hold in holds:
        t=hold['k'][0].shape[0]
        ix=torch.linspace(0,t-1,min(t,32)).long()
        raw=[m.dequantize_nvfp4(hold['q'][0][ix],hold['q'][1][ix]).float()]
        raw.extend(m.dequantize_nvfp4(*hold[r]).float() for r in 'kv')
        target=m._joint_output(*raw,qh,kh,d,ix,query_subset=True)
        q=m._joint_quant_queries(hold,parent['q_state'],qh,kh,d,ix)
        v=m._deq_params(m.hif4_dynamic_quantize_v(*hold['v'],kh,d,parent['v_state']))
        old=m._deq_params(m._kp_parent_k(*hold['k'],kh,d,parent['k_state']))
        new=m._deq_params(m._kp_quant(raw[1],candidate,kh,d,m._KP_VARIANT))
        def loss(k):
            e=(m._joint_output(q,k,v,qh,kh,d,ix,query_subset=True)-target).square().mean((-1,-2))
            return torch.stack((e[::2].mean(0),e[1::2].mean(0)))
        ratios.append(loss(new)/loss(old).clamp_min(1e-30))
    ratios=torch.stack(ratios)
    take=(ratios.mean(0)<.995).all(0)&(ratios.amax((0,1))<1.02)
    state=parent
    if bool(take.any()):
        state=dict(parent,k_state=dict(candidate,kp_variant=m._KP_VARIANT,kp_active=take,kp_fallback=parent['k_state']))
    return state,dict(accepted=int(take.sum()),ratios=ratios.tolist(),rule='mean per query fold <0.995 and worst fold <1.02')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=('feedback','covariance'),default='feedback')
    parser.add_argument('--audit-rejections',action='store_true',help='Offline diagnostic only: evaluate rejected candidate without changing production guard.')
    parser.add_argument('--cross-calibration',action='store_true',help='Train on calibration 0/2/3; select on untouched 1/4.')
    args=parser.parse_args()
    assert not (args.audit_rejections and args.cross_calibration)
    torch.set_num_threads(4)
    dataset='local_fitted_models_attention_feedback_v2' if args.dataset=='feedback' else 'local_fitted_models_covariance_v1'
    bank=torch.load(ROOT/dataset/'attn.pt',weights_only=False)
    m=load_solution(ROOT/'solution/solution_v324.py')
    source=(ROOT/'tools/attention_translation_search_core.py').read_text().replace('ts_','kp_').replace('TS_','KP_')
    exec(compile(source,'key_phase_guard','exec'),m.__dict__)
    m.phase_search=phase_search;exec(compile(CORE,'key_phase','exec'),m.__dict__)
    captured={};original_quant=m._kp_quant
    def capture(raw,st,kh,d,variant):
        captured['state']=copy.deepcopy(st)
        return original_quant(raw,st,kh,d,variant)
    m._kp_quant=capture
    rows=[]
    for gi in ((0,1,5,14,19,49) if args.dataset=='feedback' else (0,1,24,25,26,49)):
        g=bank[gi];dims=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        train=[g['calib'][i] for i in (0,2,3)] if args.cross_calibration else g['calib']
        parent=m._kp_parent_calibration(train,*dims);snapshot=copy.deepcopy(parent)
        m.hif4_calibration_attention=lambda *_,_st=parent:copy.deepcopy(_st)
        with contextlib.redirect_stdout(io.StringIO()):old,_=evaluate_attention(m,[g])
        captured.clear()
        state=m._kp_guard(train,parent,*dims)
        guard_debug=copy.deepcopy(m.KP_DEBUG)
        if args.cross_calibration and 'state' in captured:
            candidate=copy.deepcopy(captured['state'])
            state,guard_debug=cross_guard(m,[g['calib'][i] for i in (1,4)],parent,candidate,dims)
        if args.audit_rejections and 'state' in captured:
            candidate=dict(captured['state'],kp_variant=m._KP_VARIANT,
                           kp_active=torch.ones(dims[1],dtype=torch.bool),kp_fallback=parent['k_state'])
            state=dict(parent,k_state=candidate)
        assert tree_equal(parent,snapshot) and tree_equal(parent['q_state'],state['q_state']) and tree_equal(parent['v_state'],state['v_state'])
        assert not validate_attention_calibration_result(state,'phase')
        m.hif4_calibration_attention=lambda *_,_st=state:copy.deepcopy(_st)
        with contextlib.redirect_stdout(io.StringIO()):scores,_=evaluate_attention(m,[g])
        for sample in g['test']:
            out=m.hif4_dynamic_quantize_k(*sample['k'],dims[1],dims[2],state['k_state'])
            assert not validate_hif4_params(out,sample['k'][0].shape,'k')
        row=dict(group=gi,baseline=old,scores=scores,delta=sum(scores)-sum(old),debug=guard_debug)
        rows.append(row);print(row,flush=True)
    label='cross' if args.cross_calibration else ('unguarded' if args.audit_rejections else 'screen')
    (ROOT/f'docs/key-phase-{label}-{args.dataset}.json').write_text(json.dumps(dict(complete=True,dataset=dataset,unguarded=args.audit_rejections,rows=rows,
        delta=sum(r['delta'] for r in rows),warning='Modeled activations, not online score evidence; no new standalone candidate.'),indent=2)+'\n')


if __name__=='__main__':main()
