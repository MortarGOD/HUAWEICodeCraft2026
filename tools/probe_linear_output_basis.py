#!/usr/bin/env python3
"""Compare unrestricted Q(W) output bases for Linear exact-target search."""

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
    spec = importlib.util.spec_from_file_location(f"basis_v{version}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hadamard_basis(solution, rows: int, width: int) -> torch.Tensor:
    identity = torch.eye(width, dtype=torch.float32)
    return solution._hadamard(identity, width)[:rows].contiguous()


def quantized_basis(solution, name: str, weight: torch.Tensor) -> tuple[dict, torch.Tensor]:
    rows, width = map(int, weight.shape)
    if name == "identity":
        basis = torch.eye(rows, width, dtype=torch.float32)
    elif name == "hadamard":
        basis = hadamard_basis(solution, rows, width)
    elif name == "qr":
        if rows >= width:
            basis = torch.linalg.qr(weight, mode="reduced").Q
        else:
            basis = torch.linalg.qr(weight.T, mode="reduced").Q.T
    elif name.startswith("duplicate"):
        if width != 2 * rows:
            raise ValueError(f"{name} requires width == 2 * rows")
        alpha = float(name.removeprefix("duplicate"))
        basis = torch.zeros_like(weight, dtype=torch.float32)
        identity64 = torch.eye(64, dtype=torch.float32)
        for group in range(rows // 64):
            row_start = group * 64
            column_start = group * 128
            basis[row_start:row_start + 64, column_start:column_start + 64] = identity64
            basis[row_start:row_start + 64, column_start + 64:column_start + 128] = alpha * identity64
    elif name.startswith("duphadamard"):
        if width != 2 * rows:
            raise ValueError(f"{name} requires width == 2 * rows")
        alpha = float(name.removeprefix("duphadamard"))
        basis = torch.zeros_like(weight, dtype=torch.float32)
        hadamard64 = hadamard_basis(solution, 64, 64)
        for group in range(rows // 64):
            row_start = group * 64
            column_start = group * 128
            basis[row_start:row_start + 64, column_start:column_start + 64] = hadamard64
            basis[row_start:row_start + 64, column_start + 64:column_start + 128] = alpha * hadamard64
    elif name.startswith("svddup"):
        duplicate = int(name.removeprefix("svddup"))
        rank = width - duplicate
        if duplicate <= 0 or rank <= 0 or duplicate > rank:
            raise ValueError(name)
        left, singular, _ = torch.linalg.svd(weight, full_matrices=False)
        columns = left * singular[None, :]
        basis = torch.empty_like(weight, dtype=torch.float32)
        basis[:, :rank] = columns[:, :rank]
        basis[:, rank:] = (1.0 / 16.0) * columns[:, :duplicate]
    elif name == "svd":
        left, singular, _ = torch.linalg.svd(weight, full_matrices=False)
        basis = left * singular[None, :]
    else:
        raise ValueError(name)
    params = solution._quantize_linear_values(
        basis, offsets=solution._LINEAR_WEIGHT_SCALE_OFFSETS
    )
    return params, solution._dequantize_linear_params(params).reshape_as(basis).float()


def continuous_center(target: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    rows, width = map(int, basis.shape)
    if rows <= width:
        gram = basis @ basis.T
        ridge = 1.0e-7 * gram.diagonal().mean()
        gram.diagonal().add_(ridge)
        return target @ torch.linalg.solve(gram, basis)
    gram = basis.T @ basis
    ridge = 1.0e-7 * gram.diagonal().mean()
    gram.diagonal().add_(ridge)
    return target @ basis @ torch.linalg.inv(gram)


def output_score(reference, output, baseline_mse: float) -> float:
    return (baseline_mse - float((reference - output).square().mean())) / baseline_mse


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=293)
    parser.add_argument("--datasets-dir", type=Path, default=Path("local_fitted"))
    parser.add_argument("--groups", default="")
    parser.add_argument("--tests", default="0,1,2,3,4")
    parser.add_argument("--bases", default="identity,hadamard,qr")
    args = parser.parse_args()
    selected_groups = {int(x) for x in args.groups.split(",") if x} if args.groups else None
    selected_tests = {int(x) for x in args.tests.split(",") if x}
    basis_names = [x for x in args.bases.split(",") if x]
    solution = load_solution(args.version)
    groups = torch.load(args.datasets_dir / "linear.pt", map_location="cpu", weights_only=False)
    totals = {"current": 0.0}
    totals.update({name: 0.0 for name in basis_names})
    cases = 0
    started = time.perf_counter()
    for gi, group in enumerate(groups):
        if selected_groups is not None and gi not in selected_groups:
            continue
        weight = dequantize_nvfp4(*group["weight"]).float()
        baseline_weight = dequantize_hif4(quantize_to_hif4(weight), tuple(weight.shape)).float()
        calibrated = solution.hif4_calibration_and_quantize_weight(
            *group["weight"], group["calib_activation_list"]
        )
        current_weight = dequantize_hif4(calibrated["weight_params"], tuple(weight.shape)).float()
        basis_states = {}
        for name in basis_names:
            if name in ("identity", "hadamard") and int(weight.shape[0]) > int(weight.shape[1]):
                continue
            params, basis = quantized_basis(solution, name, weight)
            state = {}
            solution._install_unrestricted_exact_target_state(
                state, weight, params, compile_projection=False
            )
            basis_states[name] = (params, basis, state)
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
            case_current = output_score(reference, current_a @ current_weight.T, baseline_mse)
            totals["current"] += case_current
            case_scores = {}
            for name, (_, basis, state) in basis_states.items():
                center = continuous_center(reference, basis)
                params = solution._quantize_linear_exact_target(
                    center, reference, basis,
                    state["unrestricted_block_gram"].float(),
                    state["unrestricted_block_inverse"].float(),
                    state["unrestricted_micro_inverse"].float(),
                    state.get("unrestricted_super_gram", None).float()
                    if state.get("unrestricted_super_gram", None) is not None else None,
                    state.get("unrestricted_super_inverse", None).float()
                    if state.get("unrestricted_super_inverse", None) is not None else None,
                    solution._LINEAR_GLOBAL_SCALE_OFFSETS,
                    task_rounding=(weight.shape[0] == weight.shape[1]),
                    wide_candidates=(weight.shape[0] != 2 * weight.shape[1]),
                    extra_refinement=(weight.shape[0] < weight.shape[1]),
                    alternate_values=activation,
                )
                quantized = solution._dequantize_linear_params(params).reshape_as(activation)
                value = output_score(reference, quantized @ basis.T, baseline_mse)
                totals[name] += value
                case_scores[name] = value
            cases += 1
            print(
                f"g={gi} t={ti} {group.get('profile')} current={case_current:.6f} "
                + " ".join(f"{name}={value:.6f}" for name, value in case_scores.items()),
                flush=True,
            )
    print(f"summary cases={cases} seconds={time.perf_counter() - started:.2f}")
    for name, value in totals.items():
        print(f"{name}={value:.9f}")


if __name__ == "__main__":
    main()
