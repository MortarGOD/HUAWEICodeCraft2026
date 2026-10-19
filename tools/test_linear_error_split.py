"""Analytic error identities and zero-side-effect production instrumentation."""
import copy
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from probe_persistent_covariance import load
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_linear_calibration_result,validate_hif4_params
from linear_error_split_core import (factor_decoder,continuous_solution,decompose,
                                     capture_dynamic,audit_original_transform,legal_search_witnesses)


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    torch.set_num_threads(4)
    rng=torch.Generator().manual_seed(2026090317)
    matrices=[torch.tensor([[1.,0.],[0.,1.],[0.,0.]]),torch.eye(3),
              torch.tensor([[1.,0.,.03125,0.],[0.,1.,0.,.03125]]),
              torch.tensor([[1.,2.],[2.,4.],[3.,6.]]),torch.zeros(3,2)]
    analytic=[]
    for b in matrices:
        y=torch.randn(5,b.shape[0],generator=rng).double()
        z=torch.randn(5,b.shape[1],generator=rng).double()
        teacher=y+torch.randn(y.shape,generator=rng).double()*.1
        geometry=factor_decoder(b)
        report,continuous=decompose(y,z,teacher,geometry)
        svd=torch.linalg.lstsq(b.double(),y.T,driver='gelsd').solution.T
        assert torch.allclose(continuous@b.double().T,svd@b.double().T,atol=1e-10,rtol=1e-10)
        assert abs(report['geometric_identity_error'])<1e-10
        assert abs(report['teacher_identity_error'])<1e-10
        assert abs(report['orthogonality_cross_mse'])<1e-10
        assert report['normal_equation_relative']<1e-10
        assert report['geometric_floor_mse']<=report['total_exact_decoder_mse']+1e-10
        analytic.append(dict(shape=list(b.shape),rank=geometry['metadata']['rank']))
    b=matrices[0]
    report,_=decompose(torch.tensor([[.3,.4,2.]],dtype=torch.float64),
                       torch.tensor([[.2,.6]],dtype=torch.float64),
                       torch.tensor([[.8,.4,1.]],dtype=torch.float64),factor_decoder(b))
    assert abs(report['geometric_floor_mse']-4/3)<1e-12
    assert abs(report['discrete_and_search_mse']-.05/3)<1e-12
    # Preserve the evaluator's BF16 output rounding rather than silently using
    # FP32 matmul. It is a signed contribution, separate from representation.
    zbf=torch.tensor([[1.3125,0]],dtype=torch.bfloat16)
    bbf=torch.tensor([[1.3125,0]],dtype=torch.bfloat16)
    exact=zbf.double()@bbf.double().T
    evaluated=zbf@bbf.T
    precision,_=decompose(exact,zbf,exact,factor_decoder(bbf),evaluated)
    assert precision['total_exact_decoder_mse']==0
    assert precision['evaluation_output_increment_mse']>0
    # A looser rank cutoff is disclosed; it does not silently become exact rank.
    truncated=factor_decoder(torch.diag(torch.tensor([1.,1e-12],dtype=torch.float64)),rtol=1e-8)
    assert truncated['metadata']['numerically_truncated'] and truncated['metadata']['rank']==1
    production=[];checks=0
    for outputs,inputs in ((128,64),(64,128)):
        m=load(316)
        w=quantize_nvfp4(torch.randn(outputs,inputs,generator=rng)*.1)
        samples=[quantize_nvfp4(torch.randn(t,inputs,generator=rng)) for t in (1,2,4,8,16)]
        calibrated=m.hif4_calibration_and_quantize_weight(*w,samples)
        assert not validate_linear_calibration_result(calibrated,(outputs,inputs),'cal')
        state=calibrated['activation_state'];saved=copy.deepcopy(state)
        packed=samples[-1];saved_input=copy.deepcopy(packed)
        original=m._quantize_linear_exact_target
        baseline=m.hif4_dynamic_quantize_activation(*packed,state)
        params,teacher,internal,route,_=capture_dynamic(m,packed,state)
        assert original is m._quantize_linear_exact_target
        assert equal(params,baseline) and equal(saved,state) and equal(saved_input,packed)
        assert not validate_hif4_params(params,packed[0].shape,'captured')
        raw=m.dequantize_nvfp4(*packed).float();weight=m.dequantize_nvfp4(*w).float()
        target=raw@weight.T
        audit=audit_original_transform(m,raw,weight,state,teacher)
        assert equal(saved_input,packed)
        if route=='direct_output':assert torch.equal(teacher,target)
        else:assert audit['captured_vs_rebuilt_teacher_mse']==0
        from hif4_quant import dequantize_hif4
        deployed=dequantize_hif4(calibrated['weight_params'],(outputs,inputs)).float()
        z=dequantize_hif4(params,tuple(raw.shape)).float()
        _,continuous=decompose(target,z,teacher,factor_decoder(deployed),z@deployed.T)
        witness,candidates=legal_search_witnesses(m,params,target,teacher,deployed,continuous,torch.arange(4),channels=8)
        assert witness['recoverable_witness_mse']>=0
        for name,candidate in candidates.items():
            assert not validate_hif4_params(candidate,(4,inputs),name)
            checks+=1
        assert equal(params,baseline) and equal(saved,state)
        # Exception propagation must also restore the original private helper.
        dynamic=m.hif4_dynamic_quantize_activation
        def fail(*args):raise RuntimeError('test exception')
        m.hif4_dynamic_quantize_activation=fail
        try:
            capture_dynamic(m,packed,state)
            raise AssertionError('Expected exception')
        except RuntimeError as e:assert str(e)=='test exception'
        finally:m.hif4_dynamic_quantize_activation=dynamic
        assert m._quantize_linear_exact_target is original
        production.append(dict(shape=[outputs,inputs],route=route,output_bit_exact=True,state_unchanged=True))
    report=dict(passed=True,analytic_cases=analytic,production=production,valid_probe_returns=checks,
                orthogonal_identity=True,teacher_identity_includes_cross_term=True,
                rank_cutoff_disclosed=True,original_helper_restored_on_error=True,
                public_outputs_preserved=True,parents_and_inputs_immutable=True,
                evaluator_dtype_rounding_separated=True)
    (ROOT/'docs/linear-error-split-invariants.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
