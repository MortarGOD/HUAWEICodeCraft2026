"""Short-state identity, long calibration, format and independent K tests."""
import ast
import copy
import hashlib
import json
import torch
from build_v323_candidate import ROOT,candidate_source
from probe_v319_value import load_solution
from test_v320_residual import same
from self_check import validate_attention_calibration_result,validate_hif4_params
from build_fitted_dataset import quantize_nvfp4


def main():
    torch.set_num_threads(4);source=candidate_source()
    p=load_solution(ROOT/'solution/solution_v322.py');m=load_solution(ROOT/'solution/solution_v322.py');other=load_solution(ROOT/'solution/solution_v322.py')
    exec(compile(source,'v323-test','exec'),m.__dict__);exec(compile(source,'v323-isolated','exec'),other.__dict__)
    old=ast.parse((ROOT/'solution/solution_v322.py').read_text()).body;new=ast.parse(source).body
    changed=[]
    for a,b in zip(old[1:],new[1:]):
        if ast.dump(a)!=ast.dump(b):changed.append(getattr(a,'name',None))
    assert len(old)==len(new) and changed==['_kt_guard','hif4_dynamic_quantize_k','_ts_quant','_ts_guard','hif4_dynamic_quantize_k'],changed
    banned=set('os pathlib io tempfile shutil subprocess socket ctypes pickle marshal importlib zipimport pkgutil runpy mmap resource pty tty sys builtins'.split())
    for node in ast.walk(ast.parse(source)):
        if isinstance(node,ast.Import):assert all(n.name.split('.')[0] not in banned for n in node.names)
        if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] not in banned
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):assert node.func.id not in ('open','exec','eval','compile','__import__')
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    checks=0;calibration=[];generator=torch.Generator().manual_seed(323)
    parent_calibration=p._kt_parent_calibration
    for gi in (4,14,48):
        group=bank[gi];qh,kh,d=[group[x] for x in ('q_num_heads','kv_num_heads','head_dim')]
        # One common v320 parent avoids independent inverse-factor rebuilds.
        shared=parent_calibration(group['calib'],qh,kh,d)
        m._kt_parent_calibration=lambda *unused,_st=shared:copy.deepcopy(_st)
        p._kt_parent_calibration=lambda *unused,_st=shared:copy.deepcopy(_st)
        st=m.hif4_calibration_attention(group['calib'],qh,kh,d);base=p.hif4_calibration_attention(group['calib'],qh,kh,d)
        assert same(st,base) and not validate_attention_calibration_result(st,'attention')
        for sample in group['test']:
            result=m.hif4_dynamic_quantize_k(*sample['k'],kh,d,st['k_state'])
            assert same(result,p.hif4_dynamic_quantize_k(*sample['k'],kh,d,st['k_state']))
            checks+=1
        for rows in (1,4096,4097,16384,16385):
            pair=quantize_nvfp4(torch.randn(rows,kh*d,generator=generator)*.3)
            saved=copy.deepcopy((pair,st['k_state']));rng=torch.get_rng_state().clone()
            result=m.hif4_dynamic_quantize_k(*pair,kh,d,st['k_state'])
            assert not validate_hif4_params(result,pair[0].shape,'k')
            assert same(result,other.hif4_dynamic_quantize_k(*pair,kh,d,copy.deepcopy(st['k_state'])))
            assert same(saved,(pair,st['k_state'])) and torch.equal(rng,torch.get_rng_state())
            if rows<=4096 or rows>16384 or pair[0].numel()>8388608:
                assert same(result,p.hif4_dynamic_quantize_k(*pair,kh,d,st['k_state']))
            checks+=1
    # Exercise element budget independently of the sequence-length budget.
    def expand(value):
        if isinstance(value,dict):return {k:(kh*2 if k=='kvh' else expand(v)) for k,v in value.items()}
        if isinstance(value,torch.Tensor) and value.ndim and value.shape[0]==kh:
            return value.repeat((2,)+(1,)*(value.ndim-1))
        return value
    wide=expand(st['k_state']);pair=quantize_nvfp4(torch.randn(8193,kh*2*d,generator=generator)*.3)
    assert pair[0].numel()>8388608 and pair[0].shape[0]<16384
    out=m.hif4_dynamic_quantize_k(*pair,kh*2,d,wide)
    assert same(out,p.hif4_dynamic_quantize_k(*pair,kh*2,d,wide))
    assert not validate_hif4_params(out,pair[0].shape,'element-budget');checks+=1
    # Fresh actual long calibration, not reuse of a short calibration state.
    for gi in (14,48):
        group=bank[gi];qh,kh,d=[group[x] for x in ('q_num_heads','kv_num_heads','head_dim')]
        samples=[{role:tuple(torch.cat([x]*4,dim=0) for x in sample[role]) for role in 'qkv'} for sample in group['calib']]
        clean=load_solution(ROOT/'solution/solution_v322.py');candidate=load_solution(ROOT/'solution/solution_v322.py')
        exec(compile(source,'long-calibration','exec'),candidate.__dict__)
        shared=clean._kt_parent_calibration(samples,qh,kh,d)
        clean._kt_parent_calibration=lambda *unused,_st=shared:copy.deepcopy(_st)
        candidate._kt_parent_calibration=lambda *unused,_st=shared:copy.deepcopy(_st)
        before=clean.hif4_calibration_attention(samples,qh,kh,d);after=candidate.hif4_calibration_attention(samples,qh,kh,d)
        assert not validate_attention_calibration_result(after,'long') and '_ts_cache' not in after
        assert same(before['q_state'],after['q_state']) and same(before['v_state'],after['v_state'])
        assert not candidate.KT_DEBUG.get('error') and not candidate.TS_DEBUG.get('error')
        output=candidate.hif4_dynamic_quantize_k(*samples[-1]['k'],kh,d,after['k_state'])
        assert not validate_hif4_params(output,samples[-1]['k'][0].shape,'long_key')
        calibration.append(dict(group=gi,rows=samples[-1]['k'][0].shape[0],kt=copy.deepcopy(candidate.KT_DEBUG),ts=copy.deepcopy(candidate.TS_DEBUG)))
        checks+=2
    report=dict(passed=True,checks=checks,sha256=hashlib.sha256(source.encode()).hexdigest(),changed_symbols=changed,
                source_security_audit=True,short_state_and_params_equal=True,independent_modules=True,inputs_rng_unchanged=True,
                long_calibration=calibration)
    (ROOT/'docs/v323-long-coverage-invariants.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


if __name__=='__main__':main()
