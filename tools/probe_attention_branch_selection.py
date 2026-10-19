"""Measure deployment-faithful Q/K branch headroom before a new candidate."""
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.verify_attention_campaign import role_calibration
from tools.check_role_decoy_calibration import role_samples
from hif4_quant import dequantize_nvfp4,dequantize_hif4,quantize_to_hif4


def variants(st,role):
    out=[('parent',st)]
    if not isinstance(st,dict):return out
    plain={k:v for k,v in st.items() if k not in ('ts_variant','ts_fallback','kt_mode')}
    out.append(('rtn',dict(plain,gq=0,rf=0)))
    if 'u' in st:out.append(('gptq',dict(plain,gq=1,rf=0)))
    if 'H' in st:out.append(('rtn_refine',dict(plain,gq=0,rf=max(1,int(st.get('rf') or 0)))))
    return out


@torch.inference_mode()
def run():
    torch.set_num_threads(4)
    m=load(437,'branch_probe')
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    report={'completed':False,'groups':[],'note':'Diagnostic only, no online score estimate. Whole-role branch selection using actual public forward; shared fixed states.'}
    path=ROOT/'docs/attention-branch-selection-probe.json'
    for isolated in (False,True):
        for gi in (0,24,49):
            group=bank[gi];qh,kh,d=[group[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
            states=role_calibration(m,group,isolated,'block_resample')
            choices={r:variants(states[r+'_state'],r) for r in 'qk'}
            def losses(sample,role):
                raw={r:dequantize_nvfp4(*sample[r]).float() for r in 'qkv'}
                ix=torch.linspace(0,len(raw['q'])-1,min(32,len(raw['q']))).long()
                target=m._joint_output(raw['q'],raw['k'],raw['v'],qh,kh,d,ix)
                decoded={r:dequantize_hif4(getattr(m,'hif4_dynamic_quantize_'+r)(*sample[r],qh if r=='q' else kh,d,states[r+'_state']),raw[r].shape) for r in 'qkv'}
                base={r:dequantize_hif4(quantize_to_hif4(raw[r]),raw[r].shape) for r in 'qkv'}
                bm=float((m._joint_output(base['q'],base['k'],base['v'],qh,kh,d,ix)-target).square().mean())
                values=[]
                for name,st in choices[role]:
                    trial=dict(decoded)
                    if name!='parent':trial[role]=dequantize_hif4(getattr(m,'hif4_dynamic_quantize_'+role)(*sample[role],qh if role=='q' else kh,d,st),raw[role].shape)
                    err=(m._joint_output(trial['q'],trial['k'],trial['v'],qh,kh,d,ix)-target).square().mean((-1,-2))
                    values.append([float(err[::2].mean()),float(err[1::2].mean())] if len(err)>1 else [float(err.mean())]*2)
                return torch.tensor(values),bm
            record={'group':gi,'isolated':isolated,'roles':{}}
            for role in 'qk':
                cal=role_samples(group['calib'],role,'block_resample',829+ord(role)) if isolated else group['calib']
                hold,bm=losses(cal[-1],role)
                eligible=(hold<hold[0]*.995).all(-1)
                eligible[0]=True
                risk=hold.mean(-1);risk[~eligible]=float('inf')
                pick=int(risk.argmin())
                tests=[]
                for sample in group['test'][:3]:
                    err,bm=losses(sample,role);avg=err.mean(-1)
                    tests.append({'mse':avg.tolist(),'baseline':bm,'selected_delta':float((avg[0]-avg[pick])/max(bm,1e-30)),
                                  'oracle_delta':float((avg[0]-avg.min())/max(bm,1e-30))})
                record['roles'][role]={'branches':[n for n,_ in choices[role]],'calibration_mse':hold.tolist(),'selected':pick,'tests':tests}
            report['groups'].append(record);path.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({'group':gi,'isolated':isolated,'selection':{r:v['branches'][v['selected']] for r,v in record['roles'].items()},
                'delta':{r:sum(t['selected_delta'] for t in v['tests']) for r,v in record['roles'].items()}}),flush=True)
    report['completed']=True;path.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':run()
