"""Small counterexamples for a document review; no submission code is changed."""
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hif4_quant import quantize_to_hif4, dequantize_hif4, dequantize_nvfp4
from self_check import validate_hif4_params, validate_frozen_state


def main():
    torch.set_num_threads(1)
    record = {}
    # Equal NVFP4 scales are not enough: each HiF4 four-element microgroup
    # has one uniform signed-code grid with positive integer codes 1..7.
    sample = torch.tensor([.5,1.,4.,6.],dtype=torch.float64)
    sf = torch.tensor([2.**e*(1+m/4) for e in range(-48,16)
                       for m in range(4) if 2.**e*(1+m/4)<=49152],dtype=torch.float64)
    step = (sf[:,None]*torch.tensor([1.,2.,4.],dtype=torch.float64)[None]/4).flatten()
    values = torch.round(sample[None]/step[:,None]).clamp(-7,7)*step[:,None]
    errors = (sample[None]-values).square().sum(1)
    best = int(errors.argmin())
    assert len(sf)==255 and float(errors[best])>0
    record['equal_nvfp4_scale_not_lossless'] = {
        'nvfp4_values_in_one_microgroup':sample.tolist(),'all_nvfp4_scales':1.,
        'enumerated_legal_sf':len(sf),'enumerated_microproducts':[1,2,4],
        'best_decoded':values[best].tolist(),'minimum_sse':float(errors[best]),
        'explanation':'Equal scales meet the claimed ratio condition, but max/min=12 exceeds the HiF4 positive code ratio 7.'}

    x = torch.tensor([[-.5,1.25,.25,1.75]*16])
    p = quantize_to_hif4(x)
    assert not validate_hif4_params(p,x.shape,'valid')
    broken = dict(p,scale_lv2=torch.full_like(p['scale_lv2'],1.1))
    failure = validate_hif4_params(broken,x.shape,'bad')
    assert failure
    record['api_contract'] = {'legal_signs_observed':p['sign'].unique().tolist(),
        'legal_mants_observed':p['mant'].unique().tolist(),'continuous_lv2_rejected':failure,
        'rank16_tensor_state_errors':validate_frozen_state({'basis':torch.eye(16)},'rank16')}
    assert not record['api_contract']['rank16_tensor_state_errors']

    v,low,high=.01,0.,.25
    document = high if (v-low)/(high-low)<.5 else low
    nearest = low if (v-low)/(high-low)<.5 else high
    assert (v-document)**2>(v-nearest)**2
    record['rounding_rule_reversed'] = {'input':v,'grid':[low,high],'document_output':document,
        'nearest_output':nearest,'document_sq_error':(v-document)**2,'nearest_sq_error':(v-nearest)**2}

    q=torch.tensor([[1.,2.],[3.,4.],[5.,6.]],dtype=torch.float64)
    k=q.flip(0)
    r=torch.tensor([[1.,1.],[1.,-1.]],dtype=torch.float64)/2**.5
    exact=(q@r)@(k@r).T
    assert torch.allclose(exact,q@k.T,rtol=1e-14,atol=1e-14)
    try:
        wrong=q@r@k.T@r.T
    except RuntimeError as e:
        wrong=str(e)
    else:
        raise AssertionError('Expected dimension mismatch in the documented order')
    record['rotation_order']={'correct_max_abs_error':float((exact-q@k.T).abs().max()),
                              'document_order_failure':wrong}

    raw=torch.tensor([1.,-1.],dtype=torch.float64)
    parent=torch.tensor([1.,0.],dtype=torch.float64)
    candidate=torch.tensor([1.25,-1.],dtype=torch.float64)
    probability=torch.tensor([.9,.1],dtype=torch.float64)
    tensor=[float((z-raw).square().mean()) for z in (parent,candidate)]
    output=[float((probability@(z-raw)).square()) for z in (parent,candidate)]
    assert tensor[1]<tensor[0] and output[1]>output[0]
    record['lower_tensor_mse_not_output_guarantee']={'raw':raw.tolist(),'parent':parent.tolist(),
        'candidate':candidate.tolist(),'p':probability.tolist(),'tensor_mse':tensor,'output_mse':output}

    # API decoder has a mandated BF16 rounding, even though later Attention is FP32.
    carrier=torch.ones(1,16);scale=torch.tensor([[1.001]])
    decoded=dequantize_nvfp4(carrier,scale)
    record['nvfp4_decode_semantics']={'dtype':str(decoded.dtype),'fp32_product':float(scale[0,0]),
                                     'api_value':float(decoded[0,0])}
    out=ROOT/'docs/attention-qwen-counterexamples.json'
    out.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
