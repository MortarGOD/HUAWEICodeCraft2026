"""Research only: variable-projection native HiF4 decoder search."""
import argparse
import copy
import json
import time
import torch
from probe_v319_value import ROOT, load_solution
from hif4_quant import dequantize_hif4, quantize_to_hif4
from self_check import validate_hif4_params, validate_linear_calibration_result


def projection(y, b):
    h = b.T @ b
    h.diagonal().add_(1e-5 * h.diagonal().mean().clamp_min(1e-20))
    z = torch.linalg.solve(h, (y @ b).T).T
    return z, y - z @ b.T


def candidates(m, parent, y):
    anchor = m._linear_score_decode(parent['weight_params']).float()
    result = []
    for rate in (.25, 1.):
        b = anchor.clone()
        latent = anchor.clone()
        losses = []
        for step in range(1, 5):
            z, residual = projection(y, b)
            h = z.T @ z
            h.diagonal().add_(.01 * h.diagonal().mean().clamp_min(1e-20))
            direction = torch.linalg.solve(h, z.T @ residual).T
            latent = latent + rate * direction
            params = m._quantize_linear_values(latent, offsets=(-1, 0, 1, 2, 3))
            b = m._linear_score_decode(params).float()
            _, after = projection(y, b)
            losses.append(float(after.square().mean()))
            if step in (1, 4):
                result.append((f'r{rate}_n{step}', params, losses.copy()))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups', default='0,4,6,7,9')
    p.add_argument('--output', default='docs/v325-subspace-screen.json')
    a = p.parse_args()
    torch.set_num_threads(4)
    m = load_solution(ROOT / 'solution/solution_v324.py')
    bank = torch.load(ROOT / a.dataset / 'linear.pt', weights_only=False)
    reports = []
    for gi in map(int, a.groups.split(',')):
        start = time.perf_counter()
        g = bank[gi]
        w = m.dequantize_nvfp4(*g['weight']).float()
        assert w.shape[0] > w.shape[1]
        bw = dequantize_hif4(quantize_to_hif4(w), tuple(w.shape)).float()
        parent = m.hif4_calibration_and_quantize_weight(*g['weight'], g['calib_activation_list'])

        def loss(packed, result):
            params = m.hif4_dynamic_quantize_activation(*packed, result['activation_state'])
            assert not validate_hif4_params(params, packed[0].shape, 'activation')
            x = m.dequantize_nvfp4(*packed).float()
            y = x @ w.T
            b = m._linear_score_decode(result['weight_params']).float()
            z = m._linear_score_decode(params).float()
            return (y - z @ b.T).square().mean(-1)

        teacher = parent['activation_state']['unrestricted_target_weight'].float()
        ys = []
        for packed in g['calib_activation_list'][:-1]:
            raw = m.dequantize_nvfp4(*packed).float()
            x = m._linear_transform(raw, parent['activation_state'])
            pair = parent['activation_state']['unrestricted_pair_transform'].float()
            x = torch.einsum('rgc,gcd->rgd', x.reshape(-1, pair.shape[0], pair.shape[-1]), pair).reshape_as(x)
            xb = dequantize_hif4(quantize_to_hif4(raw), tuple(raw.shape)).float()
            bm = ((raw @ w.T) - xb @ bw.T).square().mean().clamp_min(1e-20)
            ix = torch.linspace(0, len(x)-1, min(len(x), 128)).long()
            ys.append((x[ix] @ teacher.T) / (len(ix) * bm).sqrt())
        options = [('parent', parent, [])]
        for metric, y in [('uniform', teacher.T), ('calibrated', torch.cat(ys))]:
            for name, params, history in candidates(m, parent, y):
                state = copy.deepcopy(parent['activation_state'])
                state.pop('unrestricted_continuous_projection', None)
                state.pop('linear_full_gram', None)
                m._install_unrestricted_exact_target_state(state, teacher, params)
                b = state['unrestricted_deployed_weight'].float()
                state['linear_full_gram'] = b.T @ b
                result = dict(weight_params=params, activation_state=state)
                assert not validate_linear_calibration_result(result, w.shape, 'candidate')
                options.append((metric + '_' + name, result, history))
        guards = []
        for name, result, history in options:
            e = loss(g['calib_activation_list'][-1], result)
            guards.append(dict(name=name, folds=[float(e[j::2].mean()) for j in (0, 1)], geometry=history))
        accepted = [i for i in range(1, len(options)) if all(guards[i]['folds'][j] < guards[0]['folds'][j] * .999 for j in (0, 1))]
        chosen = min(accepted, key=lambda i: sum(guards[i]['folds'])) if accepted else 0
        cases = []
        for ti, packed in enumerate(g['test_activation_list']):
            x = m.dequantize_nvfp4(*packed).float()
            xb = dequantize_hif4(quantize_to_hif4(x), tuple(x.shape)).float()
            bm = float(((x @ w.T) - xb @ bw.T).square().mean())
            old = loss(packed, parent).mean()
            new = loss(packed, options[chosen][1]).mean() if chosen else old
            cases.append(dict(test=ti, parent=1-float(old)/bm, selected=1-float(new)/bm, delta=float(old-new)/bm))
        item = dict(group=gi, chosen=options[chosen][0], guards=guards, cases=cases, delta=sum(c['delta'] for c in cases), research_seconds=time.perf_counter()-start)
        reports.append(item)
        print(gi, item['chosen'], item['delta'], item['research_seconds'], flush=True)
    (ROOT / a.output).write_text(json.dumps(dict(complete=True, dataset=a.dataset, groups=reports, delta=sum(r['delta'] for r in reports), score_prediction_allowed=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
