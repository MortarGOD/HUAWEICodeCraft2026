#!/usr/bin/env python3
"""Probe recycling low-sensitivity Q(W) blocks as residual code digits."""

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
from probe_joint_als_weight import install, search, transform


def load_solution(version: int):
    path = ROOT / "solution" / f"solution_v{version}.py"
    spec = importlib.util.spec_from_file_location(f"recycled_v{version}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=293)
    parser.add_argument("--datasets-dir", type=Path, default=Path("local_fitted"))
    parser.add_argument("--groups", default="0,1,2,3,4,5,6,9")
    parser.add_argument("--tests", default="0,1,2,3,4")
    parser.add_argument("--counts", default="1,2")
    parser.add_argument("--alpha", type=float, default=0.0625)
    args = parser.parse_args()
    selected_groups = {int(x) for x in args.groups.split(",") if x}
    selected_tests = {int(x) for x in args.tests.split(",") if x}
    counts = [int(x) for x in args.counts.split(",") if x]
    solution = load_solution(args.version)
    groups = torch.load(args.datasets_dir / "linear.pt", map_location="cpu", weights_only=False)
    totals = {"current": 0.0}
    totals.update({f"recycle{count}": 0.0 for count in counts})
    started = time.perf_counter()
    for gi, group in enumerate(groups):
        if gi not in selected_groups:
            continue
        weight = dequantize_nvfp4(*group["weight"]).float()
        baseline_weight = dequantize_hif4(quantize_to_hif4(weight), tuple(weight.shape)).float()
        calibrated = solution.hif4_calibration_and_quantize_weight(
            *group["weight"], group["calib_activation_list"]
        )
        parent_state = calibrated["activation_state"]
        target_weight = parent_state["unrestricted_target_weight"].float()
        parent_weight = parent_state["unrestricted_deployed_weight"].float()
        shape = int(parent_state["linear_shape"])
        calibration = torch.cat([
            transform(solution, dequantize_nvfp4(*packed).float(), parent_state)
            for packed in group["calib_activation_list"]
        ])
        blocks = int(target_weight.shape[1]) // 64
        activation_energy = calibration.reshape(-1, blocks, 64).square().mean(0)
        weight_energy = target_weight.reshape(-1, blocks, 64).square().sum(0)
        sensitivity = (activation_energy * weight_energy).sum(-1)
        order = sensitivity.argsort()
        candidates = {}
        for count in counts:
            low = order[:count]
            high = order[-count:].flip(0)
            basis = parent_weight.clone().reshape(-1, blocks, 64)
            basis[:, low] = args.alpha * basis[:, high]
            basis = basis.reshape_as(parent_weight)
            params = solution._quantize_linear_values(
                basis, offsets=solution._LINEAR_WEIGHT_SCALE_OFFSETS
            )
            candidate_weight = solution._dequantize_linear_params(params).reshape_as(basis).float()
            candidate_state = install(solution, target_weight, params)
            candidates[count] = (candidate_weight, candidate_state, low.tolist(), high.tolist())
        print(
            f"g={gi} {group.get('profile')} sensitivity_ratio="
            f"{float(sensitivity.min()/sensitivity.max().clamp_min(1e-30)):.4g}", flush=True
        )
        for ti, packed in enumerate(group["test_activation_list"]):
            if ti not in selected_tests:
                continue
            raw = dequantize_nvfp4(*packed).float()
            reference = raw @ weight.T
            baseline_a = dequantize_hif4(quantize_to_hif4(raw), tuple(raw.shape)).float()
            baseline_mse = float((reference - baseline_a @ baseline_weight.T).square().mean())
            parent_params = solution.hif4_dynamic_quantize_activation(*packed, parent_state)
            parent_a = dequantize_hif4(parent_params, tuple(raw.shape)).float()
            parent_output_weight = dequantize_hif4(calibrated["weight_params"], tuple(weight.shape)).float()
            current = (baseline_mse - float((reference - parent_a @ parent_output_weight.T).square().mean())) / baseline_mse
            totals["current"] += current
            values = transform(solution, raw, parent_state)
            target = values @ target_weight.T
            case = {}
            for count, (candidate_weight, candidate_state, _, _) in candidates.items():
                qa = search(solution, values, target, candidate_state, shape)
                value = (baseline_mse - float((target - qa @ candidate_weight.T).square().mean())) / baseline_mse
                totals[f"recycle{count}"] += value
                case[count] = value
            print(
                f"  t={ti} current={current:.6f} "
                + " ".join(f"r{count}={case[count]:.6f}" for count in counts), flush=True
            )
    print(f"summary seconds={time.perf_counter() - started:.2f}")
    for name, value in totals.items():
        print(f"{name}={value:.9f}")


if __name__ == "__main__":
    main()
