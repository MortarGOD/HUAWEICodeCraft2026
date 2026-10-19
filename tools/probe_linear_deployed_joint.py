"""Joint fixed-decoder research using actual public dynamic calls and scorer dtype.

Only calibration data selects candidates. No submission file is changed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import load_solution
from hif4_quant import dequantize_nvfp4, dequantize_hif4, quantize_to_hif4
from self_check import validate_linear_calibration_result, validate_hif4_params


def tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def choose_guard(rows):
    eligible = [i for i, row in enumerate(rows) if all(
        row['folds'][fold] <= rows[0]['folds'][fold] * (1.-1e-4)
        for fold in range(2))]
    return min(eligible, key=lambda i: rows[i]['loss']) if eligible else 0


def residual_update(z, y, b, ridge):
    """Ridge residual fit; dual system avoids a large input-width factorization."""
    r = y - z @ b.T
    kernel = z @ z.T
    lam = ridge * kernel.diagonal().mean().clamp_min(1e-30)
    kernel.diagonal().add_(lam)
    u = torch.linalg.solve(kernel, z)
    correction = r.T @ u
    relative = (kernel @ u - z).norm() / z.norm().clamp_min(1e-30)
    return correction, float(relative)


@torch.inference_mode()
def run(args):
    source = ROOT / 'solution/solution_v315.py'
    module = load_solution(source)
    dataset = ROOT / args.dataset
    if not dataset.name.startswith('local_fitted_models') or 'archive' in dataset.parts:
        raise ValueError('Current checkpoint-derived data only')
    bank = torch.load(dataset / 'linear.pt', weights_only=False, map_location='cpu')
    report = dict(parent=315, dataset=args.dataset, complete=False, groups=[],
                  fit_rows=args.fit_rows, inverse_rounds=args.inverse_rounds,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  score_prediction_allowed=False,
                  provenance='Real checkpoint projections, modeled activations, reduced dimensions',
                  selection='First four calibration sets fit; last set even/odd rows guard; test never selects')
    if args.full_shape:
        from build_model_proxy_dataset import MODELS, build_linear_group
        layers = [(model, layer) for model in MODELS for layer in model.linear_layers]
        for gi in map(int, args.groups.split(',')):
            model, layer = layers[gi]
            cache = ROOT / 'local_model_tensor_cache'
            cache_path = cache / f"{model.name}--{layer.name.replace('.', '_')}.pt"
            if not cache_path.exists():
                raise FileNotFoundError(f'Local cached weights required: {cache_path}')
            bank[gi] = build_linear_group(model, layer, 2026090317 + gi, 'online-shape',
                                         (1, 10, 128, 512, 1024), 'checkpoint', cache)
        report['provenance'] = 'Unreduced model/TP-local cached checkpoint projections; modeled activations, NOT actual online data'
        report['full_shape_seed_base'] = 2026090317
    started = time.perf_counter()
    for gi in map(int, args.groups.split(',')):
        group = bank[gi]
        w = dequantize_nvfp4(*group['weight']).float()
        if w.shape[0] <= w.shape[1]:
            raise ValueError('This research targets up only')
        t = time.perf_counter()
        parent = module.hif4_calibration_and_quantize_weight(
            *group['weight'], group['calib_activation_list'])
        parent_calibration_seconds = time.perf_counter() - t
        errors = validate_linear_calibration_result(parent, w.shape, 'joint.parent')
        if errors:
            raise AssertionError(errors)
        pstate = parent['activation_state']
        saved_parent = copy.deepcopy(parent)
        saved_calibration = copy.deepcopy(group['calib_activation_list'])
        bp = dequantize_hif4(parent['weight_params'], tuple(w.shape))
        validations = 1

        def forward(packed, calibrated):
            nonlocal validations
            a = dequantize_nvfp4(*packed)
            params = module.hif4_dynamic_quantize_activation(*packed, calibrated['activation_state'])
            errs = validate_hif4_params(params, tuple(a.shape), 'joint.dynamic')
            if errs:
                raise AssertionError(errs)
            validations += 1
            z = dequantize_hif4(params, tuple(a.shape))
            b = dequantize_hif4(calibrated['weight_params'], tuple(w.shape))
            y = a.float() @ w.T
            output = z @ b.T
            return z, y, (y - output.float()).square().mean(-1)

        fit_z, fit_y = [], []
        for packed in group['calib_activation_list'][:-1]:
            z, y, _ = forward(packed, parent)
            n = min(args.fit_rows, z.shape[0])
            idx = torch.linspace(0, z.shape[0]-1, n).round().long()
            # Each layout has equal weight despite its different token count/energy.
            scale = (n * y.square().mean()).clamp_min(1e-30).rsqrt()
            fit_z.append(z.float()[idx] * scale)
            fit_y.append(y[idx] * scale)
        z = torch.cat(fit_z)
        y = torch.cat(fit_y)
        correction, solve_relative = residual_update(z, y, bp.float(), .1)
        candidates = [('parent', parent)]
        for blend in (.5, 1.):
            params = module._quantize_linear_values(
                bp.float() + blend * correction, offsets=module._LINEAR_WEIGHT_SCALE_OFFSETS)
            state = dict(pstate)
            state.pop('unrestricted_continuous_projection', None)
            module._install_unrestricted_exact_target_state(
                state, pstate['unrestricted_target_weight'].float(), params)
            calibrated = dict(weight_params=params, activation_state=state)
            errors = validate_linear_calibration_result(calibrated, w.shape, 'joint.candidate')
            if errors:
                raise AssertionError(errors)
            validations += 1
            candidates.append((f'als_ridge.1_blend{blend}', calibrated))
        # Exact inverse counterpart of the existing first-order Neumann correction.
        encoder = pstate['unrestricted_continuous_projection'].float()
        proposal = torch.linalg.solve(encoder, pstate['unrestricted_target_weight'].float().T).T
        params = module._quantize_linear_values(proposal, offsets=module._LINEAR_WEIGHT_SCALE_OFFSETS)
        state = dict(pstate)
        state.pop('unrestricted_continuous_projection', None)
        module._install_unrestricted_exact_target_state(
            state, pstate['unrestricted_target_weight'].float(), params)
        inverse = dict(weight_params=params, activation_state=state)
        errors = validate_linear_calibration_result(inverse, w.shape, 'joint.inverse')
        if errors:
            raise AssertionError(errors)
        validations += 1
        candidates.append(('inverse_encoder', inverse))
        for iteration in range(2, args.inverse_rounds + 1):
            encoder = state['unrestricted_continuous_projection'].float()
            proposal = torch.linalg.solve(encoder, pstate['unrestricted_target_weight'].float().T).T
            params = module._quantize_linear_values(proposal, offsets=module._LINEAR_WEIGHT_SCALE_OFFSETS)
            state = dict(pstate)
            state.pop('unrestricted_continuous_projection', None)
            module._install_unrestricted_exact_target_state(
                state, pstate['unrestricted_target_weight'].float(), params)
            candidate = dict(weight_params=params, activation_state=state)
            errors = validate_linear_calibration_result(candidate, w.shape, 'joint.inverse.iteration')
            if errors:
                raise AssertionError(errors)
            validations += 1
            candidates.append((f'inverse_encoder_{iteration}', candidate))
        guard = group['calib_activation_list'][-1]
        rows = []
        for name, candidate in candidates:
            _, _, error = forward(guard, candidate)
            rows.append(dict(name=name, loss=float(error.mean()),
                             folds=[float(error[i::2].mean()) for i in range(2)]))
        chosen = choose_guard(rows)
        # Freeze selection BEFORE accessing any test activations.
        selected = candidates[chosen][1]
        candidate_seconds = time.perf_counter() - t
        cases = []
        baseline_w = dequantize_hif4(quantize_to_hif4(w), tuple(w.shape)).float()
        for ti, packed in enumerate(group['test_activation_list']):
            a = dequantize_nvfp4(*packed).float()
            reference = a @ w.T
            base_a = dequantize_hif4(quantize_to_hif4(a), tuple(a.shape)).float()
            base_mse = float((reference - base_a @ baseline_w.T).square().mean())
            _, _, error = forward(packed, parent)
            parent_score = 1 - float(error.mean()) / base_mse
            if chosen:
                _, _, error = forward(packed, selected)
            score = 1 - float(error.mean()) / base_mse
            cases.append(dict(test=ti, rows=int(a.shape[0]), parent=parent_score,
                              selected=score, delta=score-parent_score))
        item = dict(index=gi, shape=list(w.shape), candidates=rows, selected=rows[chosen]['name'],
                    source_model=group.get('source_model'), source_tensor=group.get('source_tensor'),
                    parent_calibration_seconds=parent_calibration_seconds,
                    solve_relative=solve_relative, calibration_research_seconds=candidate_seconds,
                    validated_returns=validations, cases=cases,
                    delta=sum(c['delta'] for c in cases))
        assert tree_equal(parent, saved_parent), 'Parent output/state mutated'
        assert tree_equal(group['calib_activation_list'], saved_calibration), 'Calibration input mutated'
        item['parent_and_calibration_unchanged'] = True
        report['groups'].append(item)
        (ROOT / args.output).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(item), flush=True)
    report.update(complete=True, seconds=time.perf_counter()-started,
                  delta=sum(g['delta'] for g in report['groups']))
    if hashlib.sha256(source.read_bytes()).hexdigest() != report['source_sha256']:
        raise AssertionError('Parent source changed')
    (ROOT / args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(delta=report['delta'], seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='local_fitted_models_attention_feedback_v2')
    parser.add_argument('--groups', default='0,6')
    parser.add_argument('--fit-rows', type=int, default=64)
    parser.add_argument('--inverse-rounds', type=int, default=1)
    parser.add_argument('--full-shape', action='store_true')
    parser.add_argument('--output', default='docs/linear-deployed-joint-research.json')
    run(parser.parse_args())
