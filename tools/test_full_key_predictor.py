"""Full-feature regression, GQA logit fusion and calibration isolation tests."""
import copy
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from full_key_predictor import fit_key_predictor,predict_keys,fuse_queries,query_probabilities
from self_check import validate_frozen_state


def main():
    torch.set_num_threads(4)
    rng=torch.Generator().manual_seed(31973)
    transform=torch.randn(64,32,generator=rng)*.1
    samples=[]
    for rows in (128,192,93):
        v=torch.randn(rows,64,generator=rng)
        samples.append(dict(v=v,k=v@transform+.15))
    original=copy.deepcopy(samples);global_rng=torch.random.get_rng_state().clone()
    state=fit_key_predictor(samples[:-1],1e-5)
    assert not validate_frozen_state(state,'predictor')
    pred=predict_keys(samples[-1]['v'],state)
    mse=float((pred-samples[-1]['k']).square().mean())
    assert mse<1e-7
    queries=torch.randn(4,11,16,generator=rng)
    fused=fuse_queries(state,queries,2,16)
    assert not validate_frozen_state(fused,'fused')
    actual=query_probabilities(samples[-1]['v'],fused)
    keys=pred.reshape(-1,2,16).transpose(0,1).repeat_interleave(2,0)
    expected=torch.softmax(queries@keys.transpose(1,2)/4,-1).reshape(2,22,93)
    fusion_error=float((actual-expected).abs().max())
    assert torch.allclose(actual,expected,atol=2e-7,rtol=1e-5)
    assert torch.equal(global_rng,torch.random.get_rng_state())
    for current,saved in zip(samples,original):
        assert all(torch.equal(current[r],saved[r]) for r in ('v','k'))
    samples[-1]['k']=torch.zeros_like(samples[-1]['k'])
    again=fit_key_predictor(samples[:-1],1e-5)
    assert torch.equal(state['coef'],again['coef'])
    identical=[dict(v=s['v'],k=s['v']) for s in samples]
    exact=fit_key_predictor(identical[:-1]);assert exact['identity']
    assert torch.equal(predict_keys(identical[-1]['v'],exact),identical[-1]['v'])
    constant=fit_key_predictor([dict(v=torch.ones(8,64),k=torch.full((8,32),2.))])
    assert not validate_frozen_state(constant,'constant')
    assert torch.equal(predict_keys(torch.ones(3,64),constant),torch.full((3,32),2.))
    result=dict(passed=True,heldout_mse=mse,fusion_error=fusion_error,state_valid=True,holdout_excluded=True,
                identity_exact=True,constant_case=True,global_rng_unchanged=True,calibration_immutable=True)
    (Path(__file__).resolve().parents[1]/'docs/full-key-predictor-invariants.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
