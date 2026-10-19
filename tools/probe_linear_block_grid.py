#!/usr/bin/env python3
"""Probe an exact high-residual block grid after the current Linear solver."""

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
from probe_linear_scalar_cd import transform


def load_solution(version: int):
    path = ROOT / "solution" / f"solution_v{version}.py"
    spec = importlib.util.spec_from_file_location(f"block_grid_v{version}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def refine(
    solution, params: dict[str, torch.Tensor], target: torch.Tensor,
    deployed: torch.Tensor, block_gram: torch.Tensor, block_inverse: torch.Tensor,
    block_limit: int, sweeps: int, offsets: tuple[int, ...],
    steps: tuple[float, ...], thresholds: tuple[float, ...],
) -> dict[str, torch.Tensor]:
    result = {name: tensor.clone() for name, tensor in params.items()}
    width = int(deployed.shape[1])
    blocks = width // 64
    rows = target.numel() // target.shape[-1]
    current = solution._dequantize_linear_params(result).reshape(rows, width).float()
    residual = target.reshape(rows, -1).float() - current @ deployed.T
    for _ in range(sweeps):
        full_gradient = residual @ deployed
        priority = []
        for block in range(blocks):
            start = block * 64
            stop = start + 64
            direction = full_gradient[:, start:stop] @ block_inverse[block]
            priority.append(float((full_gradient[:, start:stop] * direction).sum()))
        order = sorted(range(blocks), key=priority.__getitem__, reverse=True)[:min(block_limit, blocks)]
        for block in order:
            start = block * 64
            stop = start + 64
            old = current[:, start:stop]
            deployed_block = deployed[:, start:stop]
            gradient = residual @ deployed_block
            direction = gradient @ block_inverse[block]
            candidate_params = []
            candidate_values = [old]
            for step in steps:
                proposal = old + step * direction
                for threshold in thresholds:
                    for offset in offsets:
                        proposal_params = solution._quantize_linear_values(
                            proposal, offsets=(offset,), rounding_threshold=threshold
                        )
                        candidate_params.append(proposal_params)
                        candidate_values.append(
                            solution._dequantize_linear_params(proposal_params).reshape(rows, 64)
                        )
            bank = torch.stack(candidate_values, dim=1)
            delta = old[:, None] - bank
            change = 2.0 * (delta * gradient[:, None]).sum(-1) + torch.einsum(
                "rnc,cd,rnd->rn", delta, block_gram[block], delta
            )
            selected = change.argmin(-1)
            chosen = torch.gather(bank, 1, selected[:, None, None].expand(-1, 1, 64)).squeeze(1)
            accepted_delta = old - chosen
            current[:, start:stop] = chosen
            residual.add_(accepted_delta @ deployed_block.T)
            for name, tensor in result.items():
                tail = tensor.shape[target.ndim:]
                view = tensor.reshape(rows, blocks, *tail)
                old_meta = view[:, block:block + 1]
                meta_bank = torch.stack(
                    [old_meta] + [candidate[name] for candidate in candidate_params], dim=1
                )
                tail2 = meta_bank.shape[3:]
                index = selected.reshape(rows, 1, 1, *([1] * len(tail2))).expand(
                    rows, 1, 1, *tail2
                )
                view[:, block:block + 1].copy_(torch.gather(meta_bank, 1, index).squeeze(1))
    return result


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=293)
    parser.add_argument("--datasets-dir", type=Path, default=Path("local_fitted"))
    parser.add_argument("--groups", default="0,6")
    parser.add_argument("--tests", default="0,1,2,3,4")
    parser.add_argument("--limits", default="2,4,8")
    parser.add_argument("--sweeps", type=int, default=1)
    parser.add_argument("--offsets", default="-2,-1,0,1,2,3,4")
    parser.add_argument("--steps", default="0.5,1,1.5")
    parser.add_argument("--thresholds", default="0.375,0.5,0.625")
    args = parser.parse_args()
    selected_groups = {int(x) for x in args.groups.split(",") if x}
    selected_tests = {int(x) for x in args.tests.split(",") if x}
    limits = [int(x) for x in args.limits.split(",") if x]
    offsets = tuple(int(x) for x in args.offsets.split(",") if x)
    steps = tuple(float(x) for x in args.steps.split(",") if x)
    thresholds = tuple(float(x) for x in args.thresholds.split(",") if x)
    solution = load_solution(args.version)
    groups = torch.load(args.datasets_dir / "linear.pt", map_location="cpu", weights_only=False)
    totals = {"current": 0.0}
    totals.update({f"grid{limit}": 0.0 for limit in limits})
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
                refined = refine(
                    solution, params, target, deployed,
                    state["unrestricted_block_gram"].float(),
                    state["unrestricted_block_inverse"].float(),
                    limit, args.sweeps, offsets, steps, thresholds,
                )
                qa = solution._dequantize_linear_params(refined).reshape_as(values).float()
                value = (baseline_mse - float((target - qa @ deployed.T).square().mean())) / baseline_mse
                totals[f"grid{limit}"] += value
                scores[limit] = value
            print(
                f"g={gi} t={ti} {group.get('profile')} current={current_score:.6f} "
                + " ".join(f"grid{limit}={scores[limit]:.6f}" for limit in limits), flush=True
            )
    print(f"summary seconds={time.perf_counter() - started:.2f}")
    for name, value in totals.items():
        print(f"{name}={value:.9f}")


if __name__ == "__main__":
    main()
