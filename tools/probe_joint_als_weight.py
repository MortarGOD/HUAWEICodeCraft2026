#!/usr/bin/env python3
"""Probe alternating least-squares Q(W)-Q(A) updates on fitted Linear groups."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hif4_quant import dequantize_hif4, dequantize_nvfp4, quantize_to_hif4


def load_solution(version: int):
    path = ROOT / "solution" / f"solution_v{version}.py"
    spec = importlib.util.spec_from_file_location(f"joint_als_v{version}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transform(solution, values: torch.Tensor, state: dict) -> torch.Tensor:
    result = solution._linear_transform(values, state)
    pair = state.get("unrestricted_pair_transform")
    if pair is not None:
        width = int(pair.shape[-1])
        result = torch.einsum(
            "rgc,gcd->rgd", result.float().reshape(-1, pair.shape[0], width), pair.float()
        ).reshape_as(result)
    basis = state.get("activation_pair_basis")
    if basis is not None:
        basis = basis.float()
        result = result + ((result @ basis) * state["activation_pair_delta"].float()) @ basis.T
    block = state.get("activation_block_transform")
    if block is not None:
        result = torch.einsum(
            "rbc,bcd->rbd", result.reshape(-1, block.shape[0], 64), block.float()
        ).reshape_as(result)
    post = state.get("post_multiplier")
    if post is not None:
        result = result * post.float()
    return result.float()


def install(solution, target_weight: torch.Tensor, params: dict) -> dict:
    state = {}
    solution._install_unrestricted_exact_target_state(state, target_weight, params)
    return state


def search(solution, values: torch.Tensor, target: torch.Tensor, state: dict, shape: int):
    projection = state.get("unrestricted_continuous_projection")
    center = values @ projection.float() if projection is not None else values
    result = solution._quantize_linear_exact_target(
        center, target, state["unrestricted_deployed_weight"].float(),
        state["unrestricted_block_gram"].float(), state["unrestricted_block_inverse"].float(),
        state["unrestricted_micro_inverse"].float(),
        state.get("unrestricted_super_gram", None).float()
        if state.get("unrestricted_super_gram", None) is not None else None,
        state.get("unrestricted_super_inverse", None).float()
        if state.get("unrestricted_super_inverse", None) is not None else None,
        solution._LINEAR_GLOBAL_SCALE_OFFSETS,
        task_rounding=(shape == 0), wide_candidates=(shape != 1),
        extra_refinement=(shape == -1), alternate_values=values if shape == 1 else None,
    )
    return solution._dequantize_linear_params(result).reshape_as(values).float()


def objective(parts: list[torch.Tensor], targets: list[torch.Tensor], weight: torch.Tensor) -> tuple[float, torch.Tensor]:
    errors = torch.stack([
        (target - activation @ weight.T).square().mean()
        for activation, target in zip(parts, targets)
    ])
    return float(errors.mean()), errors


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=293)
    parser.add_argument("--datasets-dir", type=Path, default=Path("local_fitted"))
    parser.add_argument("--groups", default="0,6")
    parser.add_argument("--tests", default="0,1,2,3,4")
    parser.add_argument("--ridges", default="1e-4,1e-3,1e-2,1e-1,1")
    parser.add_argument("--blends", default="0.25,0.5,0.75,1")
    parser.add_argument(
        "--rows-per-layout", type=int, default=0,
        help="uniformly sample at most this many calibration rows per layout",
    )
    parser.add_argument(
        "--sketch-rows", type=int, default=0,
        help="Gaussian-sketch all calibration rows before the ALS solve",
    )
    args = parser.parse_args()
    selected_groups = {int(x) for x in args.groups.split(",") if x}
    selected_tests = {int(x) for x in args.tests.split(",") if x}
    ridges = [float(x) for x in args.ridges.split(",") if x]
    blends = [float(x) for x in args.blends.split(",") if x]
    solution = load_solution(args.version)
    groups = torch.load(args.datasets_dir / "linear.pt", map_location="cpu", weights_only=False)
    total_current = total_candidate = total_oracle = 0.0
    started = time.perf_counter()
    for gi, group in enumerate(groups):
        if gi not in selected_groups:
            continue
        raw_weight = dequantize_nvfp4(*group["weight"]).float()
        baseline_weight = dequantize_hif4(quantize_to_hif4(raw_weight), tuple(raw_weight.shape)).float()
        calibrated = solution.hif4_calibration_and_quantize_weight(
            *group["weight"], group["calib_activation_list"]
        )
        parent_state = calibrated["activation_state"]
        target_weight = parent_state["unrestricted_target_weight"].float()
        parent_params = calibrated["weight_params"]
        parent_weight = solution._dequantize_linear_params(parent_params).reshape_as(target_weight).float()
        shape = -1 if target_weight.shape[0] < target_weight.shape[1] else (
            1 if target_weight.shape[0] > target_weight.shape[1] else 0
        )
        calibration_values = []
        for packed in group["calib_activation_list"]:
            part = transform(
                solution, dequantize_nvfp4(*packed).float(), parent_state
            )
            if args.rows_per_layout and int(part.shape[0]) > args.rows_per_layout:
                indices = torch.linspace(
                    0, int(part.shape[0]) - 1, args.rows_per_layout
                ).round().to(torch.int64)
                part = part.index_select(0, indices)
            calibration_values.append(part)
        calibration_targets = [part @ target_weight.T for part in calibration_values]
        parent_exact = install(solution, target_weight, parent_params)
        parent_qa = [
            search(solution, part, target, parent_exact, shape)
            for part, target in zip(calibration_values, calibration_targets)
        ]
        flat_a = torch.cat(parent_qa)
        flat_target = torch.cat(calibration_targets)
        residual = flat_target - flat_a @ parent_weight.T
        if args.sketch_rows and args.sketch_rows < int(flat_a.shape[0]):
            generator = torch.Generator().manual_seed(20260903 + gi)
            sketch = torch.randn(
                args.sketch_rows, int(flat_a.shape[0]), generator=generator
            ) / (args.sketch_rows ** 0.5)
            flat_a = sketch @ flat_a
            residual = sketch @ residual
        use_primal = int(flat_a.shape[0]) > int(flat_a.shape[1])
        if use_primal:
            kernel = flat_a.T @ flat_a
            cross = residual.T @ flat_a
        else:
            kernel = flat_a @ flat_a.T
            cross = None
        diagonal_mean = kernel.diagonal().mean().clamp_min(torch.finfo(torch.float32).tiny)
        identity = torch.eye(int(kernel.shape[0]), dtype=torch.float32)
        candidate_rows = []
        parent_loss, parent_layout = objective(parent_qa, calibration_targets, parent_weight)
        candidate_rows.append((parent_loss, 0.0, 0.0, parent_params, parent_weight, parent_exact, parent_layout))
        for ridge in ridges:
            regularized = kernel + ridge * diagonal_mean * identity
            correction = (
                torch.linalg.solve(regularized, cross.T).T
                if use_primal
                else residual.T @ torch.linalg.solve(regularized, flat_a)
            )
            for blend in blends:
                proposal = parent_weight + blend * correction
                params = solution._quantize_linear_values(
                    proposal, offsets=solution._LINEAR_WEIGHT_SCALE_OFFSETS
                )
                deployed = solution._dequantize_linear_params(params).reshape_as(target_weight).float()
                state = install(solution, target_weight, params)
                qa = [
                    search(solution, part, target, state, shape)
                    for part, target in zip(calibration_values, calibration_targets)
                ]
                loss, layout = objective(qa, calibration_targets, deployed)
                candidate_rows.append((loss, ridge, blend, params, deployed, state, layout))
        candidate_rows.sort(key=lambda row: row[0])
        selected = next(
            row for row in candidate_rows
            if bool((row[6] <= 1.02 * parent_layout).all())
        )
        print(
            f"g={gi} {group.get('profile')} parent_cal={parent_loss:.9g} "
            f"selected_cal={selected[0]:.9g} ridge={selected[1]:g} blend={selected[2]:g}",
            flush=True,
        )
        test_sums = [0.0 for _ in candidate_rows]
        for ti, packed in enumerate(group["test_activation_list"]):
            if ti not in selected_tests:
                continue
            raw = dequantize_nvfp4(*packed).float()
            reference = raw @ raw_weight.T
            baseline_a = dequantize_hif4(quantize_to_hif4(raw), tuple(raw.shape)).float()
            baseline_mse = float((reference - baseline_a @ baseline_weight.T).square().mean())
            current_params = solution.hif4_dynamic_quantize_activation(*packed, parent_state)
            current_a = dequantize_hif4(current_params, tuple(raw.shape)).float()
            current_score = (baseline_mse - float((reference - current_a @ dequantize_hif4(
                parent_params, tuple(raw_weight.shape)
            ).float().T).square().mean())) / baseline_mse
            values = transform(solution, raw, parent_state)
            target = values @ target_weight.T
            scores = []
            for candidate_index, row in enumerate(candidate_rows):
                qa = search(solution, values, target, row[5], shape)
                candidate_score = (
                    baseline_mse
                    - float((target - qa @ row[4].T).square().mean())
                ) / baseline_mse
                scores.append(candidate_score)
                test_sums[candidate_index] += candidate_score
            selected_index = next(
                index for index, row in enumerate(candidate_rows) if row is selected
            )
            selected_score = scores[selected_index]
            oracle_score = max(scores)
            total_current += current_score
            total_candidate += selected_score
            total_oracle += oracle_score
            print(
                f"  t={ti} current={current_score:.6f} selected={selected_score:.6f} "
                f"oracle={oracle_score:.6f}", flush=True
            )
        print("  valid fixed-QW test ranking:", flush=True)
        for index in sorted(range(len(candidate_rows)), key=test_sums.__getitem__, reverse=True)[:6]:
            row = candidate_rows[index]
            print(
                f"    total={test_sums[index]:.9f} cal={row[0]:.9g} "
                f"ridge={row[1]:g} blend={row[2]:g}", flush=True
            )
    print(
        f"summary seconds={time.perf_counter() - started:.2f} current={total_current:.9f} "
        f"selected={total_candidate:.9f} oracle={total_oracle:.9f}", flush=True
    )


if __name__ == "__main__":
    main()
