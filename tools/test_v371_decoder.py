"""Continuous decoder identity including partial residual-stage blocks."""
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from self_check import validate_hif4_params,validate_linear_calibration_result
from tools.build_fitted_dataset import quantize_nvfp4


def main():
    torch.set_num_threads(4);torch.manual_seed(371)
    m=load_solution(ROOT/'solution/solution_v371.py')
    for outputs,inputs in ((64,64),(128,256),(192,512)):
        w=torch.randn(outputs,inputs)*.1
        acts=[torch.randn(n,inputs) for n in (1,10,32)]
        slots,scales=m._direct_output_layout(w,acts)
        params=m._direct_output_weight_params(outputs,inputs,slots,scales)
        assert not validate_hif4_params(params,w.shape,'decoder')
        decoded=m._dequantize_linear_params(params)
        target=torch.randn(7,outputs)
        for stage in range(slots.shape[1]):
            active=slots[:,stage]>=0;idx=torch.nonzero(active).flatten()
            local=slots[active,stage]-stage*outputs
            carrier=torch.empty(7,idx.numel())
            carrier.index_copy_(-1,local,target[:,idx]/scales[active,stage])
            carrier=m._hadamard(carrier,64)/8
            y=carrier@decoded[:,stage*outputs:stage*outputs+idx.numel()].T
            expected=torch.zeros_like(target);expected[:,idx]=target[:,idx]
            torch.testing.assert_close(y,expected,atol=3e-6,rtol=3e-6)
        wq=quantize_nvfp4(w);aq=[quantize_nvfp4(a) for a in acts]
        state=m.hif4_calibration_and_quantize_weight(*wq,aq)
        assert not validate_linear_calibration_result(state,wq[0].shape,'linear')
        for a in aq:
            p=m.hif4_dynamic_quantize_activation(*a,state['activation_state'])
            assert not validate_hif4_params(p,a[0].shape,'activation')
        print(outputs,inputs,'PASS exact continuous stages, full HiF4/state validation')


if __name__=='__main__':main()
