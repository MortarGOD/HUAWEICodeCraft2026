"""Probe the exact softmax K-translation gauge and calibrated logit gain.

For any per-head constant c, softmax(Q(K-c).T)=softmax(QK.T).
This is not a surrogate assumption about the unknown attention distribution.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import types
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import evaluate_attention
from probe_persistent_covariance import load


def configured(mode):
    source = (ROOT / 'solution/solution_v310.py').read_text()
    marker = 'def _dyn_qk(quant, scale, state, num_heads, head_dim, role=None):\n    x = dequantize_nvfp4(quant, scale).float()'
    assert source.count(marker) == 1
    source = source.replace(marker, marker + '\n    if role == "k" and isinstance(state, dict) and state.get("key_center"):\n        x = x - x.mean(0, keepdim=True)')
    m = types.ModuleType('key_gauge_experiment')
    exec(compile(source, 'key_gauge_experiment', 'exec'), m.__dict__)
    m.GPTQ_BLOCK = 32
    parent = m.hif4_calibration_attention
    m.GAUGE_DEBUG = {}
    def calibration(samples, qh, kh, d):
        st = parent(samples, qh, kh, d)
        proposed = dict(st, k_state=dict(st['k_state'], key_center=True))
        if mode == 'center_metric':
            gram = torch.zeros(kh, d, d)
            for smp in samples[:-1]:
                x = m.dequantize_nvfp4(*smp['k']).float()
                x = x - x.mean(0, keepdim=True)
                if x.shape[0] > 1024:
                    x = x[::(x.shape[0]+1023)//1024]
                if 'ks' in st['k_state']:
                    x = (x.reshape(-1, kh, d)*st['k_state']['ks'].float()[None]).reshape(x.shape)
                xb = x.reshape(-1, kh, d)
                if st['k_state'].get('rot'):
                    xb = xb @ m._make_R(d, int(st['k_state'].get('rseed') or 0))
                xb = xb.transpose(0, 1)
                gram += xb.transpose(1, 2) @ xb
            u = m._upper_cholesky_inv(gram)
            if u is not None and st['q_state'].get('gq'):
                qs = dict(st['q_state'], u=u)
                if 'H' in qs: qs['H'] = gram.to(torch.bfloat16)
                proposed['q_state'] = qs
        m.GAUGE_DEBUG.clear()
        if mode == 'center_force':
            m.GAUGE_DEBUG['accepted'] = True
            return proposed
        hold = samples[-1]
        raw = [m.dequantize_nvfp4(*hold[r]).float() for r in 'qkv']
        vp = m._deq_params(m.hif4_dynamic_quantize_v(*hold['v'], kh, d, st['v_state']))
        target = m._attention_out(*raw, qh, kh, d)
        losses = []
        for state in (st, proposed):
            q = m._deq_params(m.hif4_dynamic_quantize_q(*hold['q'], qh, d, state['q_state']))
            k = m._deq_params(m.hif4_dynamic_quantize_k(*hold['k'], kh, d, state['k_state']))
            e = (m._attention_out(q, k, vp, qh, kh, d) - target).square()
            losses.append(torch.stack([e[::2].mean(), e[1::2].mean()]))
        accept = bool((losses[1] < losses[0] * 0.999).all())
        m.GAUGE_DEBUG.update(accepted=accept, base=float(losses[0].mean()), candidate=float(losses[1].mean()))
        return proposed if accept else st
    m.hif4_calibration_attention = calibration
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups', default='0,1,2,3,4,25,26,27,28,29')
    p.add_argument('--modes', default='base,center,center_metric')
    p.add_argument('--output', default='docs/v314-key-gauge-screen.json')
    args = p.parse_args()
    bank = torch.load(ROOT / 'local_fitted_models_covariance_v1/attn.pt', weights_only=False)
    groups = [bank[int(i)] for i in args.groups.split(',')]
    results = {}
    for mode in args.modes.split(','):
        m = load(310) if mode == 'base' else configured(mode)
        m.GPTQ_BLOCK = 32
        diagnostics = []
        parent = m.hif4_calibration_attention
        def capture(*a):
            st = parent(*a)
            diagnostics.append(dict(getattr(m, 'GAUGE_DEBUG', {})))
            return st
        m.hif4_calibration_attention = capture
        with contextlib.redirect_stdout(io.StringIO()):
            scores, sec = evaluate_attention(m, groups)
        results[mode] = dict(score=sum(scores), seconds=sec, cases=scores, diagnostics=diagnostics)
        print(mode, sum(scores), sec, diagnostics, flush=True)
        (ROOT / args.output).write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__': main()
