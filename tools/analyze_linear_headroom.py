#!/usr/bin/env python3
"""Measure continuous and direct-quantization headroom behind Linear Q(A)."""

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
    spec = importlib.util.spec_from_file_location(f"headroom_v{version}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def output_mse(
    activation: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
) -> float:
    return float((target - activation @ weight.T).square().mean())


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=287)
    parser.add_argument("--datasets-dir", type=Path, default=Path("local_fitted"))
    parser.add_argument("--groups", default="")
    parser.add_argument("--tests", default="0,1,2,3,4")
    args = parser.parse_args()

    selected_groups = (
        {int(item) for item in args.groups.split(",") if item}
        if args.groups else None
    )
    selected_tests = {int(item) for item in args.tests.split(",") if item}
    solution = load_solution(args.version)
    groups = torch.load(
        args.datasets_dir / "linear.pt", map_location="cpu", weights_only=False
    )
    totals = {"baseline": 0.0, "current": 0.0, "continuous": 0.0, "rtn": 0.0}
    started = time.perf_counter()
    case_count = 0
    for group_index, group in enumerate(groups):
        if selected_groups is not None and group_index not in selected_groups:
            continue
        weight = dequantize_nvfp4(*group["weight"]).float()
        baseline_weight = dequantize_hif4(
            quantize_to_hif4(weight), tuple(weight.shape)
        ).float()
        result = solution.hif4_calibration_and_quantize_weight(
            *group["weight"], group["calib_activation_list"]
        )
        deployed_weight = dequantize_hif4(
            result["weight_params"], tuple(weight.shape)
        ).float()
        for test_index, packed in enumerate(group["test_activation_list"]):
            if test_index not in selected_tests:
                continue
            activation = dequantize_nvfp4(*packed).float()
            target = activation @ weight.T
            baseline_activation = dequantize_hif4(
                quantize_to_hif4(activation), tuple(activation.shape)
            ).float()
            current_params = solution.hif4_dynamic_quantize_activation(
                *packed, result["activation_state"]
            )
            current = dequantize_hif4(
                current_params, tuple(activation.shape)
            ).float()
            # Solve B X^T = Y^T.  This is the exact unconstrained optimum in
            # the coordinate system of the returned deployable Q(W).
            continuous = torch.linalg.lstsq(
                deployed_weight, target.T, driver="gelsy"
            ).solution.T
            direct = dequantize_hif4(
                quantize_to_hif4(continuous), tuple(continuous.shape)
            ).float()
            errors = {
                "baseline": output_mse(baseline_activation, baseline_weight, target),
                "current": output_mse(current, deployed_weight, target),
                "continuous": output_mse(continuous, deployed_weight, target),
                "rtn": output_mse(direct, deployed_weight, target),
            }
            baseline = max(errors["baseline"], 1.0e-30)
            scores = {name: (baseline - value) / baseline for name, value in errors.items()}
            for name in totals:
                totals[name] += scores[name]
            case_count += 1
            print(
                f"g={group_index} t={test_index} profile={group.get('profile')} "
                f"current={scores['current']:.6f} rtn_cont={scores['rtn']:.6f} "
                f"continuous={scores['continuous']:.6f} "
                f"gap={scores['continuous'] - scores['current']:.6f}",
                flush=True,
            )
    print(
        f"summary cases={case_count} seconds={time.perf_counter() - started:.2f} "
        + " ".join(f"{name}={value:.6f}" for name, value in totals.items()),
        flush=True,
    )


if __name__ == "__main__":
    main()
