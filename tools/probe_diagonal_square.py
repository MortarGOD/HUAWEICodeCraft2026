#!/usr/bin/env python3
"""Probe a quantized diagonal Q(W) basis for unrestricted square Linear."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hif4_quant import dequantize_hif4, dequantize_nvfp4, quantize_to_hif4


def load_solution(version: int):
    path = ROOT / "solution" / f"solution_v{version}.py"
    spec = importlib.util.spec_from_file_location(f"diagonal_v{version}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def output_mse(a: torch.Tensor, w: torch.Tensor, target: torch.Tensor) -> float:
    return float((target - a @ w.T).square().mean())


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=288)
    parser.add_argument("--datasets-dir", type=Path, default=Path("local_fitted"))
    parser.add_argument("--groups", default="")
    args = parser.parse_args()
    selected = {int(x) for x in args.groups.split(",") if x} if args.groups else None
    solution = load_solution(args.version)
    groups = torch.load(
        args.datasets_dir / "linear.pt", map_location="cpu", weights_only=False
    )
    for group_index, group in enumerate(groups):
        if selected is not None and group_index not in selected:
            continue
        weight = dequantize_nvfp4(*group["weight"]).float()
        if weight.shape[0] != weight.shape[1]:
            continue
        calibration = torch.cat([
            dequantize_nvfp4(*packed).float()
            for packed in group["calib_activation_list"]
        ])
        calibration_output = calibration @ weight.T
        output_rms = calibration_output.square().mean(0).clamp_min(1.0e-30).sqrt()
        baseline_weight = dequantize_hif4(
            quantize_to_hif4(weight), tuple(weight.shape)
        ).float()
        current_result = solution.hif4_calibration_and_quantize_weight(
            *group["weight"], group["calib_activation_list"]
        )
        current_weight = dequantize_hif4(
            current_result["weight_params"], tuple(weight.shape)
        ).float()
        totals = {"current": 0.0}
        hadamard = solution._hadamard(
            torch.eye(weight.shape[0], dtype=torch.float32),
            int(weight.shape[0]),
        )
        for alpha in (0.0, 0.25, 0.50, 0.75, 1.0):
            diagonal = output_rms.pow(alpha)
            diagonal /= diagonal.log().mean().exp()
            diagonal_weight = torch.diag(diagonal)
            diagonal_params = solution._quantize_linear_values(
                diagonal_weight, offsets=(-2, -1, 0, 1, 2, 3, 4)
            )
            deployed = solution._dequantize_linear_params(diagonal_params)
            deployed_diagonal = deployed.diagonal().clone()
            key = f"a{alpha:.2f}"
            totals[key] = 0.0
            for packed in group["test_activation_list"]:
                activation = dequantize_nvfp4(*packed).float()
                target = activation @ weight.T
                baseline_activation = dequantize_hif4(
                    quantize_to_hif4(activation), tuple(activation.shape)
                ).float()
                baseline_mse = output_mse(
                    baseline_activation, baseline_weight, target
                )
                current_params = solution.hif4_dynamic_quantize_activation(
                    *packed, current_result["activation_state"]
                )
                current = solution._dequantize_linear_params(current_params)
                current_mse = output_mse(current, current_weight, target)
                q_values = target / deployed_diagonal
                q_params = solution._quantize_linear_values(
                    q_values,
                    deployed_diagonal.square().reshape(-1, 8, 2, 4),
                    offsets=(-2, -1, 0, 1, 2, 3, 4, 5, 6),
                )
                q_values = solution._dequantize_linear_params(q_params)
                candidate_mse = output_mse(q_values, deployed, target)
                if alpha == 0.0:
                    totals["current"] += (
                        baseline_mse - current_mse
                    ) / max(baseline_mse, 1.0e-30) / 5.0
                totals[key] += (
                    baseline_mse - candidate_mse
                ) / max(baseline_mse, 1.0e-30)
            totals[key] /= 5.0
        for alpha in (0.0, 0.50, 1.0):
            row_scale = output_rms.pow(alpha)
            row_scale /= row_scale.log().mean().exp()
            basis_params = solution._quantize_linear_values(
                hadamard * row_scale[:, None],
                offsets=(-2, -1, 0, 1, 2, 3, 4),
            )
            basis = solution._dequantize_linear_params(basis_params)
            row_norm = basis.square().sum(1).clamp_min(1.0e-30)
            importance = basis.square().sum(0).reshape(-1, 8, 2, 4)
            key = f"h{alpha:.2f}"
            totals[key] = 0.0
            for packed in group["test_activation_list"]:
                activation = dequantize_nvfp4(*packed).float()
                target = activation @ weight.T
                baseline_activation = dequantize_hif4(
                    quantize_to_hif4(activation), tuple(activation.shape)
                ).float()
                baseline_mse = output_mse(
                    baseline_activation, baseline_weight, target
                )
                q_values = (target / row_norm) @ basis
                q_params = solution._quantize_linear_values(
                    q_values,
                    importance,
                    offsets=(-2, -1, 0, 1, 2, 3, 4, 5, 6),
                )
                q_values = solution._dequantize_linear_params(q_params)
                candidate_mse = output_mse(q_values, basis, target)
                totals[key] += (
                    baseline_mse - candidate_mse
                ) / max(baseline_mse, 1.0e-30)
            totals[key] /= 5.0
        print(group_index, group.get("profile"), totals, flush=True)


if __name__ == "__main__":
    main()
