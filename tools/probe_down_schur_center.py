#!/usr/bin/env python3
"""Probe exact row-space centres for unrestricted Linear down projections."""

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
    spec = importlib.util.spec_from_file_location(f"down_schur_v{version}", path)
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
            "rgc,gcd->rgd",
            result.float().reshape(-1, pair.shape[0], width),
            pair.float(),
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
    return result


def score(reference: torch.Tensor, output: torch.Tensor, baseline_mse: float) -> float:
    return (baseline_mse - float((reference - output).square().mean())) / baseline_mse


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=293)
    parser.add_argument("--datasets-dir", type=Path, default=Path("local_fitted_rectangular"))
    parser.add_argument("--tests", default="0,1,2,3,4")
    parser.add_argument("--ridges", default="0,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3")
    args = parser.parse_args()
    tests = {int(item) for item in args.tests.split(",") if item}
    ridges = [float(item) for item in args.ridges.split(",") if item]
    solution = load_solution(args.version)
    groups = torch.load(args.datasets_dir / "linear.pt", map_location="cpu", weights_only=False)
    totals = {"current": 0.0, "rtn_exact": 0.0}
    totals.update({f"schur_{ridge:g}": 0.0 for ridge in ridges})
    cases = 0
    started = time.perf_counter()
    for gi, group in enumerate(groups):
        weight = dequantize_nvfp4(*group["weight"]).float()
        if int(weight.shape[0]) >= int(weight.shape[1]):
            continue
        baseline_weight = dequantize_hif4(quantize_to_hif4(weight), tuple(weight.shape)).float()
        calibrated = solution.hif4_calibration_and_quantize_weight(
            *group["weight"], group["calib_activation_list"]
        )
        state = calibrated["activation_state"]
        deployed = state["unrestricted_deployed_weight"].float()
        target_weight = state["unrestricted_target_weight"].float()
        gram = deployed @ deployed.T
        diagonal_mean = gram.diagonal().mean()
        solves = {}
        identity = torch.eye(int(gram.shape[0]), dtype=torch.float32)
        for ridge in ridges:
            solves[ridge] = torch.linalg.solve(
                gram + ridge * diagonal_mean * identity, deployed
            )
        for ti, packed in enumerate(group["test_activation_list"]):
            if ti not in tests:
                continue
            raw = dequantize_nvfp4(*packed).float()
            reference = raw @ weight.T
            baseline_a = dequantize_hif4(quantize_to_hif4(raw), tuple(raw.shape)).float()
            baseline_mse = float((reference - baseline_a @ baseline_weight.T).square().mean())
            params = solution.hif4_dynamic_quantize_activation(*packed, state)
            current = dequantize_hif4(params, tuple(raw.shape)).float()
            totals["current"] += score(reference, current @ dequantize_hif4(
                calibrated["weight_params"], tuple(weight.shape)
            ).float().T, baseline_mse)
            values = transform(solution, raw, state).float()
            target = values @ target_weight.T
            residual = target - values @ deployed.T
            exact = values + residual @ solves[0.0]
            rtn = solution._dequantize_linear_params(
                solution._quantize_linear_values(exact, offsets=solution._LINEAR_GLOBAL_SCALE_OFFSETS)
            )
            totals["rtn_exact"] += score(target, rtn @ deployed.T, baseline_mse)
            row_scores = {}
            for ridge in ridges:
                center = values + residual @ solves[ridge]
                result = solution._quantize_linear_exact_target(
                    values, target, deployed,
                    state["unrestricted_block_gram"].float(),
                    state["unrestricted_block_inverse"].float(),
                    state["unrestricted_micro_inverse"].float(),
                    state.get("unrestricted_super_gram", None).float()
                    if state.get("unrestricted_super_gram", None) is not None else None,
                    state.get("unrestricted_super_inverse", None).float()
                    if state.get("unrestricted_super_inverse", None) is not None else None,
                    solution._LINEAR_GLOBAL_SCALE_OFFSETS,
                    task_rounding=False, wide_candidates=True, extra_refinement=True,
                    alternate_values=center,
                )
                quantized = solution._dequantize_linear_params(result).reshape_as(values)
                value = score(target, quantized @ deployed.T, baseline_mse)
                totals[f"schur_{ridge:g}"] += value
                row_scores[ridge] = value
            cases += 1
            print(
                f"g={gi} t={ti} profile={group.get('profile')} current={totals['current']:.6f} "
                f"best_ridge={max(row_scores, key=row_scores.get):g} best={max(row_scores.values()):.6f}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    print(f"summary cases={cases} seconds={elapsed:.2f}")
    for name, value in totals.items():
        print(f"{name}={value:.9f}")


if __name__ == "__main__":
    main()
