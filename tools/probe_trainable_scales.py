#!/usr/bin/env python3
"""Probe STE-trained channel scales on the current transformed Linear basis."""

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
    spec = importlib.util.spec_from_file_location(f"scales_v{version}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transform(solution, values: torch.Tensor, state: dict) -> torch.Tensor:
    result = solution._linear_transform(values, state)
    pair = state.get("unrestricted_pair_transform")
    if pair is not None:
        result = torch.einsum(
            "rgc,gcd->rgd",
            result.reshape(-1, pair.shape[0], pair.shape[-1]),
            pair.float(),
        ).reshape_as(result)
    basis = state.get("activation_pair_basis")
    if basis is not None:
        basis = basis.float()
        result = result + (
            (result @ basis) * state["activation_pair_delta"].float()
        ) @ basis.T
    block = state.get("activation_block_transform")
    if block is not None:
        result = torch.einsum(
            "rbc,bcd->rbd",
            result.reshape(-1, block.shape[0], 64),
            block.float(),
        ).reshape_as(result)
    post = state.get("post_multiplier")
    if post is not None:
        result = result * post.float()
    return result


def score(baseline: float, player: float) -> float:
    return (baseline - player) / max(baseline, 1.0e-30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=288)
    parser.add_argument("--datasets-dir", type=Path, default=Path("local_fitted"))
    parser.add_argument("--groups", default="5,6")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.03)
    args = parser.parse_args()
    solution = load_solution(args.version)
    groups = torch.load(
        args.datasets_dir / "linear.pt", map_location="cpu", weights_only=False
    )
    for group_index in [int(x) for x in args.groups.split(",") if x]:
        group = groups[group_index]
        weight = dequantize_nvfp4(*group["weight"]).float()
        result = solution.hif4_calibration_and_quantize_weight(
            *group["weight"], group["calib_activation_list"]
        )
        state = result["activation_state"]
        transformed_weight = state["unrestricted_target_weight"].float()
        current_weight = solution._dequantize_linear_params(
            result["weight_params"]
        )
        calibration = torch.cat([
            transform(solution, dequantize_nvfp4(*packed).float(), state)
            for packed in group["calib_activation_list"]
        ])
        if calibration.shape[0] > 64:
            indices = torch.linspace(
                0, calibration.shape[0] - 1, 64
            ).round().long()
            calibration = calibration.index_select(0, indices)
        output_rows = min(256, transformed_weight.shape[0])
        output_index = torch.linspace(
            0, transformed_weight.shape[0] - 1, output_rows
        ).round().long()
        sampled_weight = transformed_weight.index_select(0, output_index)
        sampled_master = current_weight.index_select(0, output_index)
        teacher = calibration @ sampled_weight.T
        log_scale = torch.zeros(
            transformed_weight.shape[1], dtype=torch.float32, requires_grad=True
        )
        optimizer = torch.optim.Adam((log_scale,), lr=args.lr)
        best_loss = float("inf")
        best_log_scale = log_scale.detach().clone()
        for step in range(args.steps):
            scale = log_scale.exp()
            activation_scaled = calibration * scale
            weight_scaled = sampled_master / scale
            with torch.no_grad():
                qa = solution._dequantize_linear_params(
                    solution._quantize_linear_values(
                        activation_scaled.detach(), offsets=(0, 1, 2, 3)
                    )
                )
                qw = solution._dequantize_linear_params(
                    solution._quantize_linear_values(
                        weight_scaled.detach(), offsets=(0, 1, 2, 3)
                    )
                )
            qa = activation_scaled + (qa - activation_scaled).detach()
            qw = weight_scaled + (qw - weight_scaled).detach()
            loss = (teacher - qa @ qw.T).square().mean()
            loss = loss + 1.0e-4 * log_scale.square().mean()
            optimizer.zero_grad()
            loss.backward()
            if float(loss.detach()) < best_loss:
                best_loss = float(loss.detach())
                best_log_scale = log_scale.detach().clone()
            optimizer.step()
            with torch.no_grad():
                log_scale.sub_(log_scale.mean()).clamp_(-1.0, 1.0)
            if step in (0, args.steps - 1):
                print(
                    f"g={group_index} step={step} loss={float(loss):.8g} "
                    f"scale=[{float(log_scale.exp().min()):.3f},"
                    f"{float(log_scale.exp().max()):.3f}]",
                    flush=True,
                )
        learned_scale = best_log_scale.exp()
        learned_weight = transformed_weight / learned_scale
        learned_params = solution._quantize_linear_values(
            current_weight / learned_scale, offsets=(0, 1, 2, 3)
        )
        learned_deployed = solution._dequantize_linear_params(learned_params)
        learned_state: dict = {}
        solution._install_unrestricted_exact_target_state(
            learned_state, learned_weight, learned_params
        )

        baseline_weight = dequantize_hif4(
            quantize_to_hif4(weight), tuple(weight.shape)
        ).float()
        current_total = 0.0
        learned_total = 0.0
        for packed in group["test_activation_list"]:
            activation = dequantize_nvfp4(*packed).float()
            transformed = transform(solution, activation, state)
            target = transformed @ transformed_weight.T
            baseline_activation = dequantize_hif4(
                quantize_to_hif4(activation), tuple(activation.shape)
            ).float()
            baseline_mse = float(
                (activation @ weight.T - baseline_activation @ baseline_weight.T)
                .square().mean()
            )
            current_params = solution.hif4_dynamic_quantize_activation(
                *packed, state
            )
            current = solution._dequantize_linear_params(current_params)
            current_mse = float((target - current @ current_weight.T).square().mean())
            values = transformed * learned_scale
            learned_a_params = solution._quantize_linear_exact_target(
                (
                    values
                    @ learned_state["unrestricted_continuous_projection"].float()
                    if "unrestricted_continuous_projection" in learned_state
                    else values
                ),
                target,
                learned_deployed,
                learned_state["unrestricted_block_gram"].float(),
                learned_state["unrestricted_block_inverse"].float(),
                learned_state["unrestricted_micro_inverse"].float(),
                learned_state.get("unrestricted_super_gram"),
                learned_state.get("unrestricted_super_inverse"),
                solution._LINEAR_GLOBAL_SCALE_OFFSETS,
                task_rounding=False,
                wide_candidates=False,
                extra_refinement=False,
                alternate_values=values,
            )
            learned_a = solution._dequantize_linear_params(learned_a_params)
            learned_mse = float((target - learned_a @ learned_deployed.T).square().mean())
            current_total += score(baseline_mse, current_mse)
            learned_total += score(baseline_mse, learned_mse)
        print(
            f"g={group_index} profile={group.get('profile')} "
            f"current={current_total:.6f} learned={learned_total:.6f} "
            f"gain={learned_total - current_total:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
