"""Deterministic regression, held-out exclusion, state and immutability checks."""
import copy
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cross_role_predictor import fit_predictor,predict_roles
from self_check import validate_frozen_state


def main():
    torch.set_num_threads(4)
    g=torch.Generator().manual_seed(316731)
    bk=torch.randn(64,32,generator=g)*.1;bv=torch.randn(64,32,generator=g)*.1
    samples=[]
    for rows in (128,192,91):
        q=torch.randn(rows,64,generator=g)
        samples.append(dict(q=q,k=q@bk+.3,v=q@bv-.2))
    original=copy.deepcopy(samples)
    rng=torch.random.get_rng_state().clone()
    state=fit_predictor(samples[:-1],'q',64,1e-5)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert not validate_frozen_state(state,'cross_role')
    predictions=predict_roles(samples[-1]['q'],state)
    errors={r:float((predictions[r]-samples[-1][r]).square().mean()) for r in 'kv'}
    assert max(errors.values())<1e-7,errors
    for s,saved in zip(samples,original):
        assert all(torch.equal(s[r],saved[r]) for r in 'qkv')
    samples[-1]['k']=torch.randn(91,32,generator=g)*100
    again=fit_predictor(samples[:-1],'q',64,1e-5)
    for key in state:
        assert torch.equal(state[key],again[key]) if isinstance(state[key],torch.Tensor) else state[key]==again[key]
    order=torch.randperm(91,generator=g)
    permuted=predict_roles(samples[-1]['q'][order],state)
    assert all(torch.allclose(permuted[r],predictions[r][order],atol=1e-6,rtol=1e-5) for r in 'kv')
    constants=[dict(q=torch.zeros(8,64),k=torch.ones(8,32),v=torch.full((8,32),2.))]
    zero_state=fit_predictor(constants,'q',64)
    assert not validate_frozen_state(zero_state,'constant')
    zero_pred=predict_roles(torch.zeros(3,64),zero_state)
    assert torch.equal(zero_pred['k'],torch.ones(3,32))
    assert torch.equal(zero_pred['v'],torch.full((3,32),2.))
    report=dict(passed=True,heldout_mse=errors,state_valid=True,calibration_immutable=True,
                holdout_labels_excluded=True,global_rng_unchanged=True,constant_case=True,
                note='Synthetic regression unit test, not empirical cross-model accuracy.')
    (ROOT/'docs/cross-role-predictor-invariants.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
