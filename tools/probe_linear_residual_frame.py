#!/usr/bin/env python3
"""Probe explicit multi-stage HiF4 residual coding with an unrestricted Q(W)."""

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
    spec = importlib.util.spec_from_file_location(f"residual_frame_v{version}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encode(solution, target: torch.Tensor, width: int, alpha: float) -> tuple[dict, torch.Tensor]:
    rows = target.numel() // target.shape[-1]
    output = int(target.shape[-1])
    output_blocks = output // 64
    copies = width // output
    flat = target.reshape(rows, output).float()
    pieces: dict[str, list[torch.Tensor]] = {}
    reconstructed = torch.zeros_like(flat)
    for block in range(output_blocks):
        residual = flat[:, block * 64:(block + 1) * 64].clone()
        for copy in range(copies):
            coefficient = alpha ** copy
            params = solution._quantize_linear_values(
                residual / coefficient,
                offsets=solution._LINEAR_GLOBAL_SCALE_OFFSETS,
            )
            quantized = solution._dequantize_linear_params(params).reshape(rows, 64)
            residual.sub_(coefficient * quantized)
            reconstructed[:, block * 64:(block + 1) * 64].add_(coefficient * quantized)
            for name, tensor in params.items():
                pieces.setdefault(name, []).append(tensor)
    used = copies * output
    if used < width:
        zeros = torch.zeros(rows, width - used, dtype=torch.float32)
        params = solution._quantize_linear_values(
            zeros, offsets=solution._LINEAR_GLOBAL_SCALE_OFFSETS
        )
        for name, tensor in params.items():
            pieces.setdefault(name, []).append(tensor)
    result = {name: torch.cat(tensors, dim=-4) for name, tensors in pieces.items()}
    return result, reconstructed.reshape_as(target)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=293)
    parser.add_argument("--datasets-dir", type=Path, default=Path("local_fitted_rectangular"))
    parser.add_argument("--groups", default="5,6,7,8,9")
    parser.add_argument("--tests", default="0,1,2,3,4")
    parser.add_argument("--alphas", default="0.25,0.125,0.0625,0.03125,0.015625,0.00390625")
    args = parser.parse_args()
    selected_groups = {int(x) for x in args.groups.split(",") if x}
    selected_tests = {int(x) for x in args.tests.split(",") if x}
    alphas = [float(x) for x in args.alphas.split(",") if x]
    solution = load_solution(args.version)
    groups = torch.load(args.datasets_dir / "linear.pt", map_location="cpu", weights_only=False)
    totals = {"current": 0.0}
    totals.update({f"alpha{alpha:g}": 0.0 for alpha in alphas})
    started = time.perf_counter()
    for gi, group in enumerate(groups):
        if gi not in selected_groups:
            continue
        weight = dequantize_nvfp4(*group["weight"]).float()
        if int(weight.shape[1]) < 2 * int(weight.shape[0]):
            continue
        baseline_weight = dequantize_hif4(quantize_to_hif4(weight), tuple(weight.shape)).float()
        calibrated = solution.hif4_calibration_and_quantize_weight(
            *group["weight"], group["calib_activation_list"]
        )
        current_weight = dequantize_hif4(calibrated["weight_params"], tuple(weight.shape)).float()
        for ti, packed in enumerate(group["test_activation_list"]):
            if ti not in selected_tests:
                continue
            activation = dequantize_nvfp4(*packed).float()
            reference = activation @ weight.T
            baseline_a = dequantize_hif4(quantize_to_hif4(activation), tuple(activation.shape)).float()
            baseline_mse = float((reference - baseline_a @ baseline_weight.T).square().mean())
            current_params = solution.hif4_dynamic_quantize_activation(
                *packed, calibrated["activation_state"]
            )
            current_a = dequantize_hif4(current_params, tuple(activation.shape)).float()
            current_score = (baseline_mse - float((reference - current_a @ current_weight.T).square().mean())) / baseline_mse
            totals["current"] += current_score
            scores = {}
            for alpha in alphas:
                _, reconstructed = encode(solution, reference, int(weight.shape[1]), alpha)
                value = (baseline_mse - float((reference - reconstructed).square().mean())) / baseline_mse
                totals[f"alpha{alpha:g}"] += value
                scores[alpha] = value
            print(
                f"g={gi} t={ti} {group.get('profile')} current={current_score:.6f} "
                + " ".join(f"a{alpha:g}={scores[alpha]:.6f}" for alpha in alphas), flush=True
            )
    print(f"summary seconds={time.perf_counter() - started:.2f}")
    for name, value in totals.items():
        print(f"{name}={value:.9f}")


if __name__ == "__main__":
    main()
