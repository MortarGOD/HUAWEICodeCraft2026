#!/usr/bin/env python3
"""Probe full-width randomized Hadamard rotations for square/up Linear."""

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
from probe_joint_als_weight import install, transform
from probe_linear_subspace_recycle import exact_search


def load_solution(version: int):
    path = ROOT / "solution" / f"solution_v{version}.py"
    spec = importlib.util.spec_from_file_location(f"global_rotation_v{version}", path)
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
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--feedback-weight", action="store_true")
    args = parser.parse_args()
    selected_groups = {int(x) for x in args.groups.split(",") if x}
    selected_tests = {int(x) for x in args.tests.split(",") if x}
    solution = load_solution(args.version)
    groups = torch.load(args.datasets_dir / "linear.pt", map_location="cpu", weights_only=False)
    total_current = total_calibrated = total_oracle = 0.0
    started = time.perf_counter()
    for gi, group in enumerate(groups):
        if gi not in selected_groups:
            continue
        weight = dequantize_nvfp4(*group["weight"]).float()
        if int(weight.shape[0]) < int(weight.shape[1]):
            continue
        width = int(weight.shape[1])
        if width & (width - 1):
            continue
        baseline_weight = dequantize_hif4(quantize_to_hif4(weight), tuple(weight.shape)).float()
        calibrated = solution.hif4_calibration_and_quantize_weight(
            *group["weight"], group["calib_activation_list"]
        )
        parent_state = calibrated["activation_state"]
        target_weight = parent_state["unrestricted_target_weight"].float()
        shape = int(parent_state["linear_shape"])
        calibration = [
            transform(solution, dequantize_nvfp4(*packed).float(), parent_state)
            for packed in group["calib_activation_list"]
        ]
        calibration_targets = [part @ target_weight.T for part in calibration]
        generator = torch.Generator().manual_seed(20260902 + gi)
        signs = [torch.ones(width)] + [
            torch.randint(0, 2, (width,), generator=generator).float().mul_(2).sub_(1)
            for _ in range(args.candidates - 1)
        ]
        variants = []
        for sign in signs:
            rotated_weight = solution._hadamard(target_weight * sign, width)
            rotated_parts = [
                solution._hadamard(part * sign, width)
                for part in calibration
            ]
            if args.feedback_weight:
                rotated_calibration = torch.cat(rotated_parts)
                activation_importance, activation_projection, activation_strength = (
                    solution._activation_output_cost_state(rotated_weight)
                )
                weight_importance, weight_projection, weight_strength = (
                    solution._activation_output_cost_state(rotated_calibration)
                )
                params = solution._feedback_quantize_weight(
                    rotated_calibration,
                    rotated_weight,
                    activation_importance,
                    activation_projection,
                    activation_strength,
                    solution._select_activation_offsets(rotated_parts),
                    weight_importance,
                    weight_projection,
                    weight_strength,
                )
            else:
                params = solution._quantize_linear_values(
                    rotated_weight, offsets=solution._LINEAR_WEIGHT_SCALE_OFFSETS
                )
            deployed = solution._dequantize_linear_params(params).reshape_as(rotated_weight).float()
            state = install(solution, target_weight, params)
            layout_errors = []
            for part, target in zip(calibration, calibration_targets):
                center = solution._hadamard(part * sign, width)
                qa_params = exact_search(solution, center, None, target, state, shape)
                qa = solution._dequantize_linear_params(qa_params).reshape_as(part).float()
                layout_errors.append((target - qa @ deployed.T).square().mean())
            variants.append((torch.stack(layout_errors).mean(), sign, deployed, state))
        calibrated_index = int(torch.stack([row[0] for row in variants]).argmin())
        print(
            f"g={gi} {group.get('profile')} cal_index={calibrated_index} "
            f"cal_losses={[round(float(row[0]), 7) for row in variants]}", flush=True
        )
        test_sums = [0.0 for _ in variants]
        for ti, packed in enumerate(group["test_activation_list"]):
            if ti not in selected_tests:
                continue
            raw = dequantize_nvfp4(*packed).float()
            reference = raw @ weight.T
            baseline_a = dequantize_hif4(quantize_to_hif4(raw), tuple(raw.shape)).float()
            baseline_mse = float((reference - baseline_a @ baseline_weight.T).square().mean())
            parent_params = solution.hif4_dynamic_quantize_activation(*packed, parent_state)
            parent_a = dequantize_hif4(parent_params, tuple(raw.shape)).float()
            parent_w = dequantize_hif4(calibrated["weight_params"], tuple(weight.shape)).float()
            current = (baseline_mse - float((reference - parent_a @ parent_w.T).square().mean())) / baseline_mse
            values = transform(solution, raw, parent_state)
            target = values @ target_weight.T
            scores = []
            for index, (_, sign, deployed, state) in enumerate(variants):
                center = solution._hadamard(values * sign, width)
                qa_params = exact_search(solution, center, None, target, state, shape)
                qa = solution._dequantize_linear_params(qa_params).reshape_as(values).float()
                score = (baseline_mse - float((target - qa @ deployed.T).square().mean())) / baseline_mse
                scores.append(score)
                test_sums[index] += score
            total_current += current
            total_calibrated += scores[calibrated_index]
            total_oracle += max(scores)
            print(
                f"  t={ti} current={current:.6f} selected={scores[calibrated_index]:.6f} "
                f"oracle={max(scores):.6f}", flush=True,
            )
        print(
            f"  fixed rotation ranking={sorted(enumerate(test_sums), key=lambda x: x[1], reverse=True)[:3]}",
            flush=True,
        )
    print(
        f"summary seconds={time.perf_counter() - started:.2f} current={total_current:.9f} "
        f"selected={total_calibrated:.9f} per_test_oracle={total_oracle:.9f}", flush=True,
    )


if __name__ == "__main__":
    main()
