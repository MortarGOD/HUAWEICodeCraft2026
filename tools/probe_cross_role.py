"""Measure held-out cross-role predictability before attempting output fitting.

Other test roles are read only for evaluation. Fits see calib[:-1] exclusively.
Output-correction diagnostics are not claimed as a legal HiF4 implementation.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cross_role_predictor import fit_predictor,predict_roles
from probe_persistent_covariance import load
from hif4_quant import dequantize_nvfp4,quantize_to_hif4
from self_check import validate_frozen_state


def decode(sample):
    return {r:dequantize_nvfp4(*sample[r]).float() for r in 'qkv'}


def mse(x,y):
    return float((x.double()-y.double()).square().mean())


def assess(m,sample,raw,states,predictor,qh,kh,d):
    source=predictor['source']
    prediction=predict_roles(raw[source],predictor)
    predicted=dict(raw)
    predicted.update({r:x.bfloat16().float() for r,x in prediction.items()})
    baseline={r:m._deq_params(quantize_to_hif4(raw[r])) for r in 'qkv'}
    player={r:m._deq_params(getattr(m,'hif4_dynamic_quantize_'+r)(*sample[r],qh if r=='q' else kh,d,states[r+'_state'])) for r in 'qkv'}
    deployed=dict(player)
    for r in prediction:
        x=predicted[r]
        # The internal dequantizer with unit scales supplies BF16 values. This
        # is an offline surrogate, not a fabricated contestant input dataset.
        pair=(x,torch.ones(x.shape[0],x.shape[1]//16))
        deployed[r]=m._deq_params(getattr(m,'hif4_dynamic_quantize_'+r)(*pair,qh if r=='q' else kh,d,states[r+'_state']))
    rows=raw['q'].shape[0]
    ix=torch.linspace(0,rows-1,min(rows,32)).long()
    output=lambda values:m._joint_output(*[values[r] for r in 'qkv'],qh,kh,d,ix)
    target=output(raw);base_error=max(mse(output(baseline),target),1e-30)
    actual_error=(output(player)-target).double()
    teacher=output(predicted)
    predicted_error=(output(deployed)-teacher).double()
    inner=float((actual_error*predicted_error).mean())
    pred_norm=float(predicted_error.square().mean())
    actual_norm=float(actual_error.square().mean())
    result=dict(rows=rows,roles={},teacher_error_over_baseline=mse(teacher,target)/base_error,
                parent_error_over_baseline=actual_norm/base_error,
                residual_cosine=inner/max((actual_norm*pred_norm)**.5,1e-30),
                predicted_residual_norm_ratio=pred_norm/max(actual_norm,1e-30),
                residual_inner=inner,pred_norm=pred_norm,actual_norm=actual_norm,baseline_mse=base_error)
    for r in prediction:
        error=mse(predicted[r],raw[r])
        result['roles'][r]=dict(nmse=error/max(float(raw[r].double().square().mean()),1e-30),
                                error_over_hif4=error/max(mse(baseline[r],raw[r]),1e-30))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups',default='1,14,26,49')
    p.add_argument('--features',default='128,256')
    p.add_argument('--sources',default='q,v')
    p.add_argument('--ridge',type=float,default=.01)
    p.add_argument('--output',default='docs/cross-role-predictability.json')
    a=p.parse_args();bank=torch.load(ROOT/a.dataset/'attn.pt',weights_only=False);m=load(314)
    report=dict(dataset=a.dataset,groups=a.groups,ridge=a.ridge,records=[],
                warning='Modeled activations. No test-label fitting. Residual subtraction is an unconstrained diagnostic, not a realizable HiF4 score.')
    for gi in map(int,a.groups.split(',')):
        g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        calibration=[decode(s) for s in g['calib']]
        states=m.hif4_calibration_attention(g['calib'],qh,kh,d)
        for source in a.sources.split(','):
            for features in map(int,a.features.split(',')):
                started=time.perf_counter()
                predictor=fit_predictor(calibration[:-1],source,features,ridge=a.ridge)
                assert not validate_frozen_state(predictor,'predictor')
                held=assess(m,g['calib'][-1],calibration[-1],states,predictor,qh,kh,d)
                alpha=max(-2.,min(2.,held['residual_inner']/max(held['pred_norm'],1e-30)))
                tests=[assess(m,s,decode(s),states,predictor,qh,kh,d) for s in g['test']]
                for result in tests:
                    result['calibrated_unconstrained_gain']=(2*alpha*result['residual_inner']-alpha*alpha*result['pred_norm'])/result['baseline_mse']
                record=dict(group=gi,source=source,features=features,effective_features=int(predictor['indices'].numel()),
                            ridge=a.ridge,hold=held,tests=tests,calibrated_alpha=alpha,
                            seconds=time.perf_counter()-started)
                report['records'].append(record)
                (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')
                print(gi,source,features,'hold teacher',round(held['teacher_error_over_baseline'],3),
                      'test teacher',round(sum(t['teacher_error_over_baseline'] for t in tests)/len(tests),3),
                      'cosine',round(sum(t['residual_cosine'] for t in tests)/len(tests),3),flush=True)


if __name__=='__main__':main()
