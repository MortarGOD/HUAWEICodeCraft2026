"""Mathematical, routing, format and calibration-isolation checks for v316."""
import ast
import copy
import json
from pathlib import Path
import sys
import torch
from probe_deployed_frames import configured, load
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_frozen_state, validate_hif4_params


def main():
    torch.set_num_threads(4)
    production = '--production' in sys.argv
    m = load(316) if production else configured()
    rng = torch.Generator().manual_seed(916316)
    qh, kh, d = 4, 2, 64
    samples = [{r: quantize_nvfp4(torch.randn(t, (qh if r=='q' else kh)*d, generator=rng))
                for r in 'qkv'} for t in (17, 24, 32)]
    gram = torch.eye(d).repeat(kh, 1, 1)
    qst = dict(kvh=kh, rot=1, rseed=0, gq=1, rf=1,
               u=m._upper_cholesky_inv(gram), H=gram.bfloat16(),
               qs=torch.ones(kh, d), temperature_gain=torch.tensor([.8,1.,1.1,1.2]))
    kst = dict(qst)
    kst.pop('qs'); kst.pop('temperature_gain'); kst['ks'] = torch.ones(kh, d)
    qst['joint_transform'] = m._make_R(d,0).expand(kh,-1,-1).contiguous()
    kst['joint_transform'] = qst['joint_transform'].clone()
    parent = dict(q_state=qst,k_state=kst,v_state=None)
    saved = copy.deepcopy(samples)
    before = torch.random.get_rng_state().clone()
    frames, labels = m._frame_candidates(samples,parent,qh,kh,d)
    assert len(frames)==3 and frames[0] is parent
    assert not validate_frozen_state(frames,'frames')
    rawq, rawk = m._frame_raw_grams(samples,qh,kh,d)
    r = m._make_R(d,0)
    expected_k = r.T @ (rawq*qst['temperature_gain'][:,None,None].square()).reshape(kh,qh//kh,d,d).sum(1) @ r
    assert torch.allclose(frames[1]['k_state']['H'].float(), expected_k, rtol=.005, atol=.015)
    assert torch.equal(frames[1]['q_state']['temperature_gain'],qst['temperature_gain'])
    altered = copy.deepcopy(samples)
    altered[-1] = {r:quantize_nvfp4(torch.randn(32,(qh if r=='q' else kh)*d,generator=rng)*20) for r in 'qkv'}
    other, _ = m._frame_candidates(altered,parent,qh,kh,d)
    for f1,f2 in zip(frames,other):
        for role in ('q_state','k_state'):
            for key,a in f1[role].items():
                b = f2[role][key]
                assert torch.equal(a,b) if isinstance(a,torch.Tensor) else a==b
    assert all(torch.equal(a,b) for s,t in zip(samples,saved) for r in 'qkv' for a,b in zip(s[r],t[r]))
    assert torch.equal(before,torch.random.get_rng_state())
    costs = torch.ones(3,2,qh)
    costs[1,:,:2]=.8; costs[2,:,2:]=.7
    assert m._frame_select(costs,kh).tolist()==[1,2]
    costs[1,1,:2]=1.1; costs[2,:,2:]=float('nan')
    assert m._frame_select(costs,kh).tolist()==[0,0]
    format_checks = 0
    subset_error = 0.
    for role,heads in (('q',qh),('k',kh)):
        original = getattr(m,'_frame_parent_'+role)
        bank = [f[role+'_state'] for f in frames]
        for rows in (1,129,1025):
            pair = quantize_nvfp4(torch.randn(rows,heads*d,generator=rng))
            choice = torch.arange(heads)%3
            state = dict(frame_bank=bank,frame_choice=choice)
            out = m._frame_dynamic(*pair,heads,d,state,original)
            assert not validate_frozen_state(state,'state')
            assert not validate_hif4_params(out,pair[0].shape,role)
            format_checks += 1
            for index in range(3):
                reference = original(*pair,heads,d,bank[0 if rows==1 else index])
                for head in range(heads):
                    if choice[head]!=index:continue
                    sl=slice(head*d//64,(head+1)*d//64)
                    assert all(torch.equal(out[k][:,sl],reference[k][:,sl]) for k in out)
            zero = dict(frame_bank=bank,frame_choice=torch.zeros(heads,dtype=torch.int64))
            p = original(*pair,heads,d,bank[0])
            z = m._frame_dynamic(*pair,heads,d,zero,original)
            assert all(torch.equal(p[k],z[k]) for k in p)
            if role=='q' and rows>1:
                ix=torch.linspace(0,rows-1,16).long()
                for st in bank:
                    full=m._deq_params(original(*pair,heads,d,st))[ix]
                    sub=m._joint_quant_queries(dict(q=pair),st,qh,kh,d,ix)
                    subset_error=max(subset_error,float((full-sub).abs().max()))
    assert subset_error==0
    calls=[]
    def counted(*args):
        calls.append(args[-1]); return m._frame_parent_q(*args)
    state=dict(frame_bank=[qst,frames[1]['q_state']],frame_choice=torch.ones(qh,dtype=torch.int64))
    m._frame_dynamic(*samples[0]['q'],qh,d,state,counted)
    assert len(calls)==1 and calls[0] is state['frame_bank'][1]
    # Reusing only local calibration results must match fresh public calls.
    context = {}
    delta = torch.tensor([-.1,.2,.05,-.04])
    guarded = m._temp_guard(samples[-1],parent,delta,qh,kh,d,context=context)
    actual_q = m._joint_quant_queries(samples[-1],guarded['q_state'],qh,kh,d,context['ix'])
    actual_k = m._deq_params(m._frame_parent_k(*samples[-1]['k'],kh,d,guarded['k_state']))
    assert torch.equal(context['q'],actual_q)
    assert torch.equal(context['k'],actual_k)
    slow = m._frame_guard(samples,guarded,qh,kh,d)
    slow_debug = copy.deepcopy(m.FRAME_DEBUG)
    fast = m._frame_guard(samples,guarded,qh,kh,d,context)
    assert m.FRAME_DEBUG == slow_debug
    assert not validate_frozen_state(fast,'fast_state')
    for role,heads in (('q',qh),('k',kh)):
        f=getattr(m,'hif4_dynamic_quantize_'+role)
        a=f(*samples[0][role],heads,d,slow[role+'_state'])
        b=f(*samples[0][role],heads,d,fast[role+'_state'])
        assert all(torch.equal(a[k],b[k]) for k in a)
    # Empty statistics never create a GPTQ candidate.
    empty=[{r:quantize_nvfp4(torch.zeros_like(s[r][0])) for r in 'qkv'} for s in samples]
    assert len(m._frame_candidates(empty,parent,qh,kh,d)[0]) == 1
    root=Path(__file__).resolve().parents[1]
    if production:
        source=(root/'solution/solution_v316.py').read_text()
        assert source==(root/'solution/solution.py').read_text()
        old=ast.parse((root/'solution/solution_v315.py').read_text())
        new=ast.parse(source)
        for a,b in zip(old.body,new.body):
            if isinstance(a,ast.Expr) and isinstance(a.value,ast.Constant) and isinstance(a.value.value,str):continue
            if isinstance(a,ast.FunctionDef) and a.name=='_temp_guard':continue
            assert ast.dump(a)==ast.dump(b),getattr(a,'name',type(a).__name__)
        banned={'os','pathlib','io','tempfile','shutil','subprocess','socket','ctypes','pickle','marshal',
                'importlib','zipimport','pkgutil','runpy','mmap','resource','pty','tty','sys','builtins'}
        for node in ast.walk(new):
            if isinstance(node,ast.Import):assert all(a.name.split('.')[0] not in banned for a in node.names)
            if isinstance(node,ast.ImportFrom):assert (node.module or '').split('.')[0] not in banned
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):assert node.func.id not in {'exec','eval','open','__import__'}
    report=dict(passed=True,format_returns=format_checks,temperature_weighted_gram=True,
                gain_retained=True,holdout_excluded=True,parent_immutable=True,
                calibration_immutable=True,global_rng_unchanged=True,mixed_routing_exact=True,
                fallback_exact=True,both_folds_required=True,nonfinite_rejected=True,
                original_T_query_subset_error=subset_error,unused_parent_call_removed=True,
                calibration_context_exact=True,context_cost_and_choice_exact=True,zero_gram_rejected=True,
                actual_standalone=production,linear_ast_frozen=production,prohibited_imports_absent=production)
    output='docs/v316-production-invariants.json' if production else 'docs/v316-invariants.json'
    (root/output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
