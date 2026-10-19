"""Paired v315 coordinate selection, cached parent; timings exclude cached fit."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import time
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from probe_persistent_covariance import load
from evaluate_local import evaluate_attention
from self_check import validate_frozen_state, validate_hif4_params


def configured():
    m = load(315)
    source = (ROOT/'solution/solution_v315.py').read_text()
    patched = temperature_context_patch(source)
    # Re-load the whole independent module namespace so function aliases match.
    exec(compile(patched, 'v315_with_calibration_context', 'exec'), m.__dict__)
    exec(compile((ROOT/'tools/deployed_frame_core.py').read_text(), 'deployed_frames', 'exec'), m.__dict__)
    return m


def temperature_context_patch(source):
    old = 'def _temp_guard(hold, states, delta, qh, kh, d):'
    assert source.count(old) == 1
    source = source.replace(old, old[:-2]+', context=None):')
    old = "    if bool(mask.any()):\n        answer['k_state']=dict(states['k_state'],key_center_mask=mask)\n    return answer"
    new = """    if bool(mask.any()):
        answer['k_state']=dict(states['k_state'],key_center_mask=mask)
    if context is not None and not _TEMP_CENTER_KEYS:
        # All query rows are independent: gather the already-selected gain.
        chosen_q = queries[0].reshape(-1,qh,d)
        for i in range(1,len(gains)):
            take = gain == gains[i]
            chosen_q = torch.where(take[None,:,None],queries[i].reshape(-1,qh,d),chosen_q)
        context.update(ix=ix,target=reference,q=chosen_q.reshape(-1,qh*d),k=k,v=v)
    return answer"""
    assert source.count(old) == 1
    return source.replace(old,new)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups', default='5,13,14,19,23,47')
    p.add_argument('--output', default='docs/v316-frames-screen.json')
    a = p.parse_args()
    bank = torch.load(ROOT/a.dataset/'attn.pt', weights_only=False)
    groups = bank if a.groups == 'all' else [bank[int(i)] for i in a.groups.split(',')]
    parents = {}
    m = configured()
    start = time.perf_counter()
    for g in groups:
        parents[id(g['calib'])] = m._frame_parent_calibration(g['calib'], *[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')])
    parent_seconds = time.perf_counter()-start
    report = dict(dataset=a.dataset, groups=a.groups, results={}, parent_seconds=parent_seconds,
                  warning='Paired cached parent. Modeled activations, no online forecast. Mode timing excludes original calibration.')
    for mode in ('parent', 'frames'):
        m = configured()
        m._frame_parent_calibration = lambda samples,*args: copy.deepcopy(parents[id(samples)])
        m._frame_calibrate_parent = lambda samples,*args: (copy.deepcopy(parents[id(samples)]), {})
        cal = m._frame_parent_calibration if mode == 'parent' else m.hif4_calibration_attention
        diags = []
        count = dict(states=0, returns=0)
        def capture(*args):
            st = cal(*args)
            assert not validate_frozen_state(st, 'frames')
            count['states'] += 1
            diags.append(copy.deepcopy(m.FRAME_DEBUG))
            return st
        m.hif4_calibration_attention = capture
        for role in 'qkv':
            original = getattr(m, 'hif4_dynamic_quantize_'+role)
            def checked(*args, _original=original, _role=role):
                out = _original(*args)
                assert not validate_hif4_params(out, args[0].shape, _role)
                count['returns'] += 1
                return out
            setattr(m, 'hif4_dynamic_quantize_'+role, checked)
        with contextlib.redirect_stdout(io.StringIO()):
            scores, seconds = evaluate_attention(m, groups)
        report['results'][mode] = dict(score=sum(scores), cases=scores, seconds=seconds, diagnostics=diags, checks=count)
        (ROOT/a.output).write_text(json.dumps(report, indent=2)+'\n')
        print(mode, sum(scores), seconds, [x.get('choice') for x in diags], flush=True)


if __name__ == '__main__':
    main()
