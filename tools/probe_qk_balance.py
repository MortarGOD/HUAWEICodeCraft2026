"""Full-matrix reciprocal balancing: Q T, K T^-T, not diagonal tuning."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import types
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import evaluate_attention
from probe_persistent_covariance import load


def configured(strength):
    source = (ROOT/'solution/solution_v310.py').read_text()
    marker = '\n    p = _dyn_table(x, None, has_scale=False)\n'
    start = source.index('def _dyn_qk(')
    end = source.index('def _dyn_v(', start)
    qk = source[start:end]
    assert qk.count(marker) == 1
    qk = qk.replace(marker, '''
    if isinstance(state, dict) and "full_balance" in state:
        t = state["full_balance"].float().repeat_interleave(num_heads // state["kvh"], 0)
        x = (x.reshape(-1, num_heads, head_dim).transpose(0, 1) @ t).transpose(0, 1).reshape(x.shape).contiguous()
''' + marker)
    source = source[:start] + qk + source[end:]
    m = types.ModuleType('full_balance_experiment')
    exec(compile(source, 'full_balance_experiment', 'exec'), m.__dict__)
    m.GPTQ_BLOCK=32
    parent=m.hif4_calibration_attention
    m.BAL_DEBUG={}

    def matrix_power(h, power):
        e,u=torch.linalg.eigh((h+h.transpose(-1,-2))*0.5)
        e=e.clamp_min(e.amax(-1)[:,None]*0.001)
        return (u*e.pow(power)[:,None,:])@u.transpose(-1,-2)

    def rawmetric(st):
        hd=torch.cholesky_inverse(st['u'].float(),upper=True)
        ridge=hd.diagonal(dim1=-2,dim2=-1).mean(-1)*(m.GPTQ_DAMP/(1+m.GPTQ_DAMP))
        return hd-ridge[:,None,None]*torch.eye(hd.shape[-1])

    def calibration(samples,qh,kh,d):
        st=parent(samples,qh,kh,d)
        m.BAL_DEBUG.clear();m.BAL_DEBUG.update(accepted=0,error=False)
        if not st['q_state'].get('gq') or not st['k_state'].get('gq'):return st
        try:
            hk,hq=rawmetric(st['q_state']),rawmetric(st['k_state'])
            a=hq/(qh//kh);b=hk
            ar=matrix_power(a,0.5);ai=matrix_power(a,-0.5)
            mm=ai@matrix_power(ar@b@ar,0.5)@ai
            t=matrix_power(mm,strength*0.5)
            e,u=torch.linalg.eigh(t);e=e.clamp(0.25,4.0)
            t=(u*e[:,None,:])@u.transpose(-1,-2)
            ti=torch.linalg.inv(t).transpose(-1,-2)
            gq=ti.transpose(-1,-2)@hk@ti
            gk=t.transpose(-1,-2)@hq@t
            uq,uk=m._upper_cholesky_inv(gq),m._upper_cholesky_inv(gk)
            if uq is None or uk is None:return st
            pq=dict(st['q_state'],full_balance=t,u=uq)
            pk=dict(st['k_state'],full_balance=ti,u=uk)
            if 'H' in pq:pq['H']=gq.to(torch.bfloat16)
            if 'H' in pk:pk['H']=gk.to(torch.bfloat16)
            proposed=dict(st,q_state=pq,k_state=pk)
            hold=samples[-1]
            raw=[m.dequantize_nvfp4(*hold[r]).float() for r in 'qkv']
            target=m._attention_out(*raw,qh,kh,d).reshape(-1,kh,qh//kh,d)
            v=m._deq_params(m.hif4_dynamic_quantize_v(*hold['v'],kh,d,st['v_state']))
            loss=[]
            for s in (st,proposed):
                q=m._deq_params(m.hif4_dynamic_quantize_q(*hold['q'],qh,d,s['q_state']))
                k=m._deq_params(m.hif4_dynamic_quantize_k(*hold['k'],kh,d,s['k_state']))
                error=(m._attention_out(q,k,v,qh,kh,d).reshape(target.shape)-target).square().mean((-1,-2))
                loss.append(torch.stack((error[::2].mean(0),error[1::2].mean(0))))
            active=(loss[1]<0.995*loss[0]).all(0)
            m.BAL_DEBUG.update(accepted=int(active.sum()),base=float(loss[0].mean()),candidate=float(loss[1].mean()),
                              invariance=float((t@ti.transpose(-1,-2)-torch.eye(d)).abs().max()))
            if not bool(active.any()):return st
            answer=dict(st)
            for role in 'qk':
                s=dict(proposed[role+'_state']);old=st[role+'_state']
                s['full_balance']=torch.where(active[:,None,None],s['full_balance'],torch.eye(d)).contiguous()
                for field in ('u','H'):
                    if field in old:s[field]=torch.where(active[:,None,None],s[field],old[field]).contiguous()
                answer[role+'_state']=s
            return answer
        except Exception:
            m.BAL_DEBUG['error']=True
            return st
    m.hif4_calibration_attention=calibration
    return m


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--groups',default='0,1,2,3,4,25,26,27,28,29')
    p.add_argument('--strengths',default='0,0.5,1')
    p.add_argument('--output',default='docs/v314-full-balance-screen.json')
    a=p.parse_args()
    bank=torch.load(ROOT/'local_fitted_models_covariance_v1/attn.pt',weights_only=False)
    groups=[bank[int(i)] for i in a.groups.split(',')]
    results={}
    for strength in map(float,a.strengths.split(',')):
        m=configured(strength) if strength else load(310);m.GPTQ_BLOCK=32
        parent=m.hif4_calibration_attention;diagnostics=[]
        def capture(*args):
            st=parent(*args);diagnostics.append(dict(getattr(m,'BAL_DEBUG',{})));return st
        m.hif4_calibration_attention=capture
        with contextlib.redirect_stdout(io.StringIO()):scores,seconds=evaluate_attention(m,groups)
        results[str(strength)]=dict(score=sum(scores),seconds=seconds,cases=scores,diagnostics=diagnostics)
        print(strength,sum(scores),seconds,diagnostics,flush=True)
        (ROOT/a.output).write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
