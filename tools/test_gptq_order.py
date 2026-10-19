"""Ordering must permute elimination only, never the returned HiF4 layout."""
import ast
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from build_fitted_dataset import quantize_nvfp4
from probe_gptq_order import configured,candidate_source
from probe_persistent_covariance import load
from self_check import validate_hif4_params,validate_attention_calibration_result


def main():
    torch.set_num_threads(4)
    m,parent=configured('conditional'),load(314)
    generator=torch.Generator().manual_seed(31641)
    qh,kh,d=4,2,64
    a=torch.randn(kh,d,d,generator=generator)
    h=a@a.transpose(-1,-2)+torch.eye(d)
    u=m._upper_cholesky_inv(h)
    state=dict(kvh=kh,rot=0,gq=1,u=u,rf=1,H=h.bfloat16())
    ordered=m._order_proposal(state)
    identity=dict(state,gptq_order=torch.arange(d).expand(kh,d),gptq_order_u=u)
    assert not validate_attention_calibration_result(dict(q_state=ordered,k_state=ordered,v_state=None),'ordered')
    inverse=u.transpose(-1,-2)@u
    ix=ordered['gptq_order']
    target=inverse.gather(1,ix[:,:,None].expand_as(inverse)).gather(2,ix[:,None,:].expand_as(inverse))
    actual=ordered['gptq_order_u'].transpose(-1,-2)@ordered['gptq_order_u']
    matrix_relative=float((actual-target).abs().max()/target.abs().max())
    assert matrix_relative<1e-5,matrix_relative
    checks=0
    for rows in (1,64,129,1025):
        sample={}
        for role,heads in (('q',qh),('k',kh)):
            pair=quantize_nvfp4(torch.randn(rows,heads*d,generator=generator));sample[role]=pair
            f=getattr(m,'hif4_dynamic_quantize_'+role)
            expected=getattr(parent,'hif4_dynamic_quantize_'+role)(*pair,heads,d,state)
            for st in (state,identity):
                got=f(*pair,heads,d,st)
                assert all(torch.equal(got[k],expected[k]) for k in got),(rows,role)
            result=f(*pair,heads,d,ordered)
            assert not validate_hif4_params(result,pair[0].shape,role)
            if rows==1:assert all(torch.equal(result[k],expected[k]) for k in result)
            checks+=1
        if rows>1:
            ix=torch.linspace(0,rows-1,min(16,rows)).long()
            full=m._deq_params(m.hif4_dynamic_quantize_q(*sample['q'],qh,d,ordered))[ix]
            subset=m._joint_quant_queries(sample,ordered,qh,kh,d,ix)
            rel=float((full-subset).square().mean().sqrt()/full.square().mean().sqrt())
            assert rel<.02,rel
    old=ast.parse((ROOT/'solution/solution_v314.py').read_text())
    new=ast.parse(candidate_source())
    for a,b in zip(old.body,new.body):
        if isinstance(a,ast.FunctionDef) and a.name in ('_dyn_qk','_joint_quant_queries'):
            continue
        assert ast.dump(a)==ast.dump(b)
    report=dict(passed=True,matrix_relative_error=matrix_relative,format_returns=checks,
                identity_exact=True,disabled_exact=True,linear_ast_unchanged=True,
                note='Offline prototype; online gain and runtime not established.')
    (ROOT/'docs/gptq-order-invariants.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
