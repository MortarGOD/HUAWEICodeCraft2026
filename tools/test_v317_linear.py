"""Production cleanup and joint-guard correctness on analytic-sized examples."""
import copy
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from build_fitted_dataset import quantize_nvfp4
from probe_linear_deployed_joint import tree_equal
from self_check import validate_linear_calibration_result, validate_hif4_params
from verify_v317_linear import source_audit


@torch.inference_mode()
def main():
    unchanged=source_audit()
    rng=torch.Generator().manual_seed(317)
    results=[]
    for o,i in ((128,64),(64,64),(64,128),(192,512),(64,512),(128,320)):
        p=load_solution(ROOT/'solution/solution_v315.py')
        n=load_solution(ROOT/'solution/solution_v317.py')
        n._LINEAR_JOINT_ROUNDS=0
        w=quantize_nvfp4(torch.randn(o,i,generator=rng)*.1)
        cal=[quantize_nvfp4(torch.randn(t,i,generator=rng)) for t in (1,2,4,8,16)]
        saved=copy.deepcopy(cal)
        a=p.hif4_calibration_and_quantize_weight(*w,cal)
        b=n.hif4_calibration_and_quantize_weight(*w,cal)
        assert tree_equal(a['weight_params'],b['weight_params'])
        assert not validate_linear_calibration_result(b,(o,i),'clean')
        for packed in cal:
            qa=p.hif4_dynamic_quantize_activation(*packed,a['activation_state'])
            qb=n.hif4_dynamic_quantize_activation(*packed,b['activation_state'])
            assert tree_equal(qa,qb)
        n._LINEAR_JOINT_ROUNDS=4
        before=copy.deepcopy(b)
        if o>i:
            result=n._linear_refine_fixed_decoder(b,*w,cal)
            assert not validate_linear_calibration_result(result,(o,i),'joint')
            assert tree_equal(b,before)
            bad=copy.deepcopy(b)
            bad['activation_state']['unrestricted_continuous_projection'].zero_()
            assert n._linear_refine_fixed_decoder(bad,*w,cal) is bad
            zeros=list(cal)
            zeros[-1]=quantize_nvfp4(torch.zeros(16,i))
            assert n._linear_refine_fixed_decoder(b,*w,zeros) is b
            for packed in cal:
                out=n.hif4_dynamic_quantize_activation(*packed,result['activation_state'])
                assert not validate_hif4_params(out,packed[0].shape,'joint')
        else:
            result=n.hif4_calibration_and_quantize_weight(*w,cal)
            assert tree_equal(result,b)
        assert tree_equal(saved,cal)
        results.append((o,i))
    print('PASS',results,'unchanged functions',unchanged,'singular solve/zero-target guard/immutability')


if __name__=='__main__':main()
