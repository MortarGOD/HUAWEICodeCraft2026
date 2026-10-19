#!/usr/bin/env python3
"""Probe systematic generator matrices for unrestricted up Q(W)."""

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
    spec = importlib.util.spec_from_file_location(f"systematic_v{version}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pivoted_rows(q: torch.Tensor) -> torch.Tensor:
    """Pure-Torch column-pivoted MGS on Q.T; returns a max-volume row set."""
    work = q.T.contiguous().clone()
    columns = int(work.shape[1])
    rank = int(work.shape[0])
    permutation = torch.arange(columns)
    norms = work.square().sum(0)
    for step in range(rank):
        pivot = step + int(norms[step:].argmax())
        if pivot != step:
            temporary = work[:, step].clone()
            work[:, step] = work[:, pivot]
            work[:, pivot] = temporary
            norm_value = norms[step].clone()
            norms[step] = norms[pivot]
            norms[pivot] = norm_value
            index_value = permutation[step].clone()
            permutation[step] = permutation[pivot]
            permutation[pivot] = index_value
        vector = work[:, step]
        vector = vector / vector.norm().clamp_min(1.0e-12)
        if step + 1 < columns:
            coefficients = vector @ work[:, step + 1:]
            work[:, step + 1:].sub_(
                vector[:, None] * coefficients[None, :]
            )
            norms[step + 1:] = work[:, step + 1:].square().sum(0)
    return permutation[:rank]


def selections(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    rows, width = map(int, weight.shape)
    result = {
        "first": torch.arange(width),
        "even": torch.arange(0, 2 * width, 2)[:width],
        "stride": torch.linspace(0, rows - 1, width).round().to(torch.int64),
    }
    q = torch.linalg.qr(weight, mode="reduced").Q
    result["pivot"] = pivoted_rows(q)
    leverage = q.square().sum(-1)
    result["leverage"] = leverage.argsort(descending=True)[:width].sort().values
    return result


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=293)
    parser.add_argument("--datasets-dir", type=Path, default=Path("local_fitted_rectangular"))
    parser.add_argument("--groups", default="0,1,2,3,4")
    parser.add_argument("--tests", default="0,1,2,3,4")
    parser.add_argument(
        "--methods", default="pivot,pivot_s05,pivot_s1",
        help="row selection, optionally suffixed by _s05 or _s1 column balance",
    )
    args = parser.parse_args()
    selected_groups = {int(x) for x in args.groups.split(",") if x}
    selected_tests = {int(x) for x in args.tests.split(",") if x}
    methods = [x for x in args.methods.split(",") if x]
    solution = load_solution(args.version)
    groups = torch.load(args.datasets_dir / "linear.pt", map_location="cpu", weights_only=False)
    totals = {"current": 0.0}
    totals.update({name: 0.0 for name in methods})
    started = time.perf_counter()
    for gi, group in enumerate(groups):
        if gi not in selected_groups:
            continue
        weight = dequantize_nvfp4(*group["weight"]).float()
        if int(weight.shape[0]) <= int(weight.shape[1]):
            continue
        baseline_weight = dequantize_hif4(quantize_to_hif4(weight), tuple(weight.shape)).float()
        calibrated = solution.hif4_calibration_and_quantize_weight(
            *group["weight"], group["calib_activation_list"]
        )
        parent_state = calibrated["activation_state"]
        target_weight = parent_state["unrestricted_target_weight"].float()
        choices = selections(target_weight)
        variants = {}
        for name in methods:
            base_name = name.split("_", 1)[0]
            index = choices[base_name]
            square = target_weight.index_select(0, index)
            condition = float(torch.linalg.cond(square))
            systematic = torch.linalg.solve(square.T, target_weight.T).T
            exponent = (
                0.5 if name.endswith("_s05")
                else (1.0 if name.endswith("_s1") else 0.0)
            )
            if exponent > 0.0:
                column_rms = systematic.square().mean(0).sqrt().clamp_min(1.0e-12)
                geometric = column_rms.log().mean().exp()
                multiplier = (geometric / column_rms).pow(exponent).clamp(
                    1.0 / 16.0, 16.0
                )
                systematic = systematic * multiplier
            params = solution._quantize_linear_values(
                systematic, offsets=solution._LINEAR_WEIGHT_SCALE_OFFSETS
            )
            deployed = solution._dequantize_linear_params(params).reshape_as(systematic).float()
            variants[name] = (deployed, install(solution, target_weight, params), condition)
        print(
            f"g={gi} {group.get('profile')} "
            + " ".join(f"{name}_cond={variants[name][2]:.3g}" for name in methods),
            flush=True,
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
            parent_w = dequantize_hif4(calibrated["weight_params"], tuple(weight.shape)).float()
            current = (baseline_mse - float((reference - parent_a @ parent_w.T).square().mean())) / baseline_mse
            totals["current"] += current
            values = transform(solution, raw, parent_state)
            target = values @ target_weight.T
            case = {}
            for name, (deployed, state, _) in variants.items():
                qa = search(solution, values, target, state, 1)
                value = (baseline_mse - float((target - qa @ deployed.T).square().mean())) / baseline_mse
                totals[name] += value
                case[name] = value
            print(
                f"  t={ti} current={current:.6f} "
                + " ".join(f"{name}={case[name]:.6f}" for name in methods), flush=True,
            )
    print(f"summary seconds={time.perf_counter() - started:.2f}")
    for name, value in totals.items():
        print(f"{name}={value:.9f}")


if __name__ == "__main__":
    main()
