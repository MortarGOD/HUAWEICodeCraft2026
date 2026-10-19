#!/usr/bin/env python3
"""Probe exact scalar coordinate descent after the current Linear solver."""

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
    spec = importlib.util.spec_from_file_location(f"scalar_cd_v{version}", path)
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


def refine(
    solution, params: dict[str, torch.Tensor], target: torch.Tensor,
    deployed: torch.Tensor, channel_limit: int, sweeps: int,
) -> dict[str, torch.Tensor]:
    result = {name: tensor.clone() for name, tensor in params.items()}
    width = int(deployed.shape[1])
    rows = target.numel() // target.shape[-1]
    current = solution._dequantize_linear_params(result).reshape(rows, width).float()
    flat_target = target.reshape(rows, -1).float()
    residual = flat_target - current @ deployed.T
    sign = result["sign"].reshape(rows, width)
    mant = result["mant"].reshape(rows, width)
    element_scale = (
        result["scale_factor"].float()
        * result["scale_lv2"].float()
        * result["scale_lv3"].float()
        * torch.ones_like(result["sign"], dtype=torch.float32)
    ).reshape(rows, width)
    diagonal = deployed.square().sum(0).clamp_min(torch.finfo(torch.float32).tiny)
    codes = torch.arange(-7, 8, dtype=torch.float32).reshape(1, 15)
    for _ in range(sweeps):
        gradient = residual @ deployed
        levels = element_scale[:, :, None] * codes.reshape(1, 1, 15) / 4.0
        delta = current[:, :, None] - levels
        changes = 2.0 * delta * gradient[:, :, None] + delta.square() * diagonal[None, :, None]
        best_change = changes.min(-1).values
        order = best_change.sum(0).argsort()[:min(channel_limit, width)].tolist()
        for channel in order:
            old = current[:, channel]
            gradient_channel = residual @ deployed[:, channel]
            candidate = element_scale[:, channel, None] * codes / 4.0
            delta_channel = old[:, None] - candidate
            change = (
                2.0 * delta_channel * gradient_channel[:, None]
                + delta_channel.square() * diagonal[channel]
            )
            selected = change.argmin(-1)
            new = torch.gather(candidate, 1, selected[:, None]).squeeze(1)
            accepted_delta = old - new
            current[:, channel] = new
            residual.add_(accepted_delta[:, None] * deployed[:, channel][None, :])
            selected_code = codes.flatten().index_select(0, selected)
            sign[:, channel] = torch.sign(selected_code).to(sign.dtype)
            mant[:, channel] = (selected_code.abs() / 4.0).to(mant.dtype)
    result["sign"] = sign.reshape_as(result["sign"])
    result["mant"] = mant.reshape_as(result["mant"])
    return result


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=293)
    parser.add_argument("--datasets-dir", type=Path, default=Path("local_fitted"))
    parser.add_argument("--groups", default="0,6")
    parser.add_argument("--tests", default="0,1,2,3,4")
    parser.add_argument("--limits", default="32,64,128,256,512")
    parser.add_argument("--sweeps", type=int, default=1)
    args = parser.parse_args()
    selected_groups = {int(x) for x in args.groups.split(",") if x}
    selected_tests = {int(x) for x in args.tests.split(",") if x}
    limits = [int(x) for x in args.limits.split(",") if x]
    solution = load_solution(args.version)
    groups = torch.load(args.datasets_dir / "linear.pt", map_location="cpu", weights_only=False)
    totals = {"current": 0.0}
    totals.update({f"cd{limit}": 0.0 for limit in limits})
    started = time.perf_counter()
    for gi, group in enumerate(groups):
        if gi not in selected_groups:
            continue
        weight = dequantize_nvfp4(*group["weight"]).float()
        baseline_weight = dequantize_hif4(quantize_to_hif4(weight), tuple(weight.shape)).float()
        calibrated = solution.hif4_calibration_and_quantize_weight(
            *group["weight"], group["calib_activation_list"]
        )
        state = calibrated["activation_state"]
        deployed = state["unrestricted_deployed_weight"].float()
        target_weight = state["unrestricted_target_weight"].float()
        for ti, packed in enumerate(group["test_activation_list"]):
            if ti not in selected_tests:
                continue
            raw = dequantize_nvfp4(*packed).float()
            reference = raw @ weight.T
            baseline_a = dequantize_hif4(quantize_to_hif4(raw), tuple(raw.shape)).float()
            baseline_mse = float((reference - baseline_a @ baseline_weight.T).square().mean())
            params = solution.hif4_dynamic_quantize_activation(*packed, state)
            values = transform(solution, raw, state)
            target = values @ target_weight.T
            current = solution._dequantize_linear_params(params).reshape_as(values).float()
            current_score = (baseline_mse - float((target - current @ deployed.T).square().mean())) / baseline_mse
            totals["current"] += current_score
            scores = {}
            for limit in limits:
                refined = refine(solution, params, target, deployed, limit, args.sweeps)
                qa = solution._dequantize_linear_params(refined).reshape_as(values).float()
                value = (baseline_mse - float((target - qa @ deployed.T).square().mean())) / baseline_mse
                totals[f"cd{limit}"] += value
                scores[limit] = value
            print(
                f"g={gi} t={ti} {group.get('profile')} current={current_score:.6f} "
                + " ".join(f"cd{limit}={scores[limit]:.6f}" for limit in limits), flush=True
            )
    print(f"summary seconds={time.perf_counter() - started:.2f}")
    for name, value in totals.items():
        print(f"{name}={value:.9f}")


if __name__ == "__main__":
    main()
