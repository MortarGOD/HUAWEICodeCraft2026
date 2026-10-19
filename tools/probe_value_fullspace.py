"""Offline capacity diagnostic. Test Q/K are never deployable V features."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.test_v422 import load
from tools.verify_attention_campaign import equal
from hif4_quant import dequantize_nvfp4, dequantize_hif4
from self_check import validate_hif4_params


@torch.inference_mode()
def search(m, x, initial, pq, target, kh, d):
    t = len(x)
    xh = x.reshape(t, kh, d).transpose(0, 1)
    y = dequantize_hif4(initial, x.shape).reshape(t, kh, d).transpose(0, 1)
    a, b = pq[:, ::2].double(), target[:, ::2].double()
    gram = a.transpose(-1, -2) @ a / a.shape[1]
    linear = a.transpose(-1, -2) @ b / a.shape[1]
    rho = gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)
    chol = torch.linalg.cholesky(gram + rho[:, None, None] * torch.eye(t))
    y = y.double(); dual = torch.zeros_like(y)
    best = initial
    best_loss = (a @ y - b).square().mean((1, 2))
    accepted = 0
    grid = tuple((off, sig) for off in range(-1, 3) for sig in m.E6M2_SIG)
    for _ in range(32):
        z = torch.cholesky_solve(linear + rho[:, None, None] * (y - dual), chol)
        proposal = (z + dual).transpose(0, 1).reshape(t, -1).float().contiguous()
        params = m._quantize_weighted(proposal, torch.ones(1, proposal.shape[1]), grid=grid)
        assert not validate_hif4_params(params, x.shape, 'fullspace')
        y = dequantize_hif4(params, x.shape).reshape(t, kh, d).transpose(0, 1).double()
        loss = (a @ y - b).square().mean((1, 2))
        take = loss < best_loss
        accepted += int(take.sum())
        block = take.repeat_interleave(d // 64)[None]
        best = {k: torch.where(block.reshape(1, -1, *([1] * (v.ndim - 2))), params[k], v)
                for k, v in best.items()}
        best_loss = torch.minimum(loss, best_loss)
        dual += z - y
    actual = dequantize_hif4(best, x.shape).reshape(t, kh, d).transpose(0, 1)
    assert torch.allclose((a @ actual.double() - b).square().mean((1, 2)), best_loss,
                          atol=1e-12, rtol=1e-6)
    return actual, accepted


@torch.inference_mode()
def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--groups', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    m = load(435, 'fullspace')
    path = ROOT / 'docs/papers/hif4-public-references/test/datasets' / args.dataset / 'attn.pt'
    bank = torch.load(path, weights_only=True)
    rows = []
    for gi in map(int, args.groups.split(',')):
        g = bank[gi]
        qh, kh, d = [g[k] for k in ('q_num_heads', 'kv_num_heads', 'head_dim')]
        state = m.hif4_calibration_attention(g['calib'], qh, kh, d)
        frozen = copy.deepcopy(state)
        for ti, s in enumerate(g['test']):
            raw = {r: dequantize_nvfp4(*s[r]).float() for r in 'qkv'}
            params = {r: getattr(m, 'hif4_dynamic_quantize_' + r)(
                *s[r], qh if r == 'q' else kh, d, state[r + '_state']) for r in 'qkv'}
            decoded = {r: dequantize_hif4(params[r], raw[r].shape) for r in 'qkv'}
            t = len(raw['v'])
            x = raw['v'].reshape(t, kh, d).transpose(0, 1)
            p = m._vr_probs(raw['q'], raw['k'], qh, kh, d).reshape(kh, -1, t)
            pq = m._vr_probs(decoded['q'], decoded['k'], qh, kh, d).reshape(kh, -1, t)
            target = p @ x
            variants = {'parent': decoded['v'].reshape(t, kh, d).transpose(0, 1)}
            for name, par in [('rtn', m._dyn_table(raw['v'], None, False)),
                              ('mean', m._vk_parent_v(*s['v'], None, kh, d))]:
                assert not validate_hif4_params(par, raw['v'].shape, name)
                variants[name] = dequantize_hif4(par, raw['v'].shape).reshape(t, kh, d).transpose(0, 1)
            counts = {}
            for name, goal in [('oracle_value', pq @ x), ('oracle_output', target)]:
                variants[name], counts[name] = search(m, raw['v'], params['v'], pq, goal, kh, d)
            row = {'group': gi, 'test': ti, 'tokens': t, 'accepted_heads_rounds': counts, 'variants': {}}
            for name, y in variants.items():
                error = pq @ y - target
                row['variants'][name] = {
                    'fit_mse': float(error[:, ::2].square().mean()),
                    'heldout_mse': float(error[:, 1::2].square().mean()),
                    'tensor_mse': float((y - x).square().mean()),
                    'changed': int((y != variants['parent']).sum())}
            rows.append(row)
        assert equal(state, frozen)
        print({'group_complete': gi}, flush=True)
    summary = {}
    for name in rows[0]['variants']:
        ratios = [r['variants'][name]['heldout_mse'] / max(r['variants']['parent']['heldout_mse'], 1e-30) for r in rows]
        summary[name] = {'mean_heldout_ratio': sum(ratios) / len(ratios),
                         'better': sum(x < 1 for x in ratios), 'worse': sum(x > 1 for x in ratios)}
    report = {'complete': True, 'dataset': args.dataset, 'data_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256((ROOT / 'solution/solution_v435.py').read_bytes()).hexdigest(),
              'cases': rows, 'summary': summary,
              'limits': 'Offline real-test-QK oracle; not deployable, not global optimum, not online estimate. Query split does not test sequence transfer. Captured traces not independently recaptured.'}
    (ROOT / args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(summary)


if __name__ == '__main__':
    run()
