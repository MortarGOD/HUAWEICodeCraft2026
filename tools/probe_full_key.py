"""Held-out V->K prediction and output sensitivity; no test fitting."""
import argparse
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from full_key_predictor import fit_key_predictor,predict_keys
from probe_persistent_covariance import load
from hif4_quant import dequantize_nvfp4,quantize_to_hif4
from self_check import validate_frozen_state


def decode(sample):return {r:dequantize_nvfp4(*sample[r]).float() for r in 'qkv'}


def assess(m,raw,predictor,qh,kh,d):
    prediction=predict_keys(raw['v'],predictor)
    base={r:m._deq_params(quantize_to_hif4(raw[r])) for r in 'qkv'}
    ix=torch.linspace(0,raw['q'].shape[0]-1,min(32,raw['q'].shape[0])).long()
    target=m._joint_output(*[raw[r] for r in 'qkv'],qh,kh,d,ix)
    standard=m._joint_output(*[base[r] for r in 'qkv'],qh,kh,d,ix)
    # True current Q/V belong to this diagnostic scorer, not the predictor.
    predicted=m._joint_output(raw['q'],prediction,raw['v'],qh,kh,d,ix)
    mse=lambda a,b:float((a.double()-b.double()).square().mean())
    key_error=mse(prediction,raw['k'])
    return dict(rows=raw['v'].shape[0],key_nmse=key_error/max(mse(raw['k'],torch.zeros_like(raw['k'])),1e-30),
                key_over_hif4=key_error/max(mse(base['k'],raw['k']),1e-30),
                output_over_hif4=mse(predicted,target)/max(mse(standard,target),1e-30))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups',default='1,5,14,26,37,49')
    p.add_argument('--ridges',default='.0001,.01')
    p.add_argument('--output',default='docs/full-key-prediction-screen.json')
    a=p.parse_args();m=load(314)
    bank=torch.load(ROOT/a.dataset/'attn.pt',weights_only=False)
    report=dict(dataset=a.dataset,records=[],warning='Modeled hidden states; current Q/V used only by the diagnostic scorer. No full-output teacher claim.')
    for gi in map(int,a.groups.split(',')):
        g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        cal=[decode(s) for s in g['calib']]
        for ridge in map(float,a.ridges.split(',')):
            fit=fit_key_predictor(cal[:-1],ridge)
            assert not validate_frozen_state(fit,'full_key')
            hold=assess(m,cal[-1],fit,qh,kh,d)
            tests=[assess(m,decode(s),fit,qh,kh,d) for s in g['test']]
            report['records'].append(dict(group=gi,ridge=ridge,identity=fit['identity'],hold=hold,tests=tests))
            (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')
            print(gi,ridge,'identity',fit['identity'],'hold K',round(hold['key_over_hif4'],3),
                  'test K',round(sum(t['key_over_hif4'] for t in tests)/len(tests),3),
                  'test PV',round(sum(t['output_over_hif4'] for t in tests)/len(tests),3),flush=True)


if __name__=='__main__':main()
