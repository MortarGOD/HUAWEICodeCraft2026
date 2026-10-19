#!/usr/bin/env python3
"""Concentrate low singular directions, then recycle one HiF4 block."""

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


def load_solution(version: int):
    path = ROOT / "solution" / f"solution_v{version}.py"
    spec = importlib.util.spec_from_file_location(f"subspace_recycle_v{version}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def minimal_subspace_rotation(low_vectors: torch.Tensor) -> torch.Tensor:
    """Map the final coordinate axes onto an orthonormal low-energy basis."""
    width, rank = map(int, low_vectors.shape)
    rotation = torch.eye(width, dtype=torch.float32)
    first = width - rank
    for index in range(rank):
        source = rotation[:, first + index]
        target = low_vectors[:, index]
        if float(source @ target) < 0.0:
            target = -target
        difference = source - target
        norm = difference.norm()
        if float(norm) <= 1.0e-7:
            continue
        reflector = difference / norm
        rotation.sub_(2.0 * reflector[:, None] * (reflector @ rotation)[None, :])
    return rotation


def exact_search(solution, center, alternate, target, state, shape):
    return solution._quantize_linear_exact_target(
        center, target, state["unrestricted_deployed_weight"].float(),
        state["unrestricted_block_gram"].float(),
        state["unrestricted_block_inverse"].float(),
        state["unrestricted_micro_inverse"].float(),
        state.get("unrestricted_super_gram", None).float()
        if state.get("unrestricted_super_gram", None) is not None else None,
        state.get("unrestricted_super_inverse", None).float()
        if state.get("unrestricted_super_inverse", None) is not None else None,
        solution._LINEAR_GLOBAL_SCALE_OFFSETS,
        task_rounding=(shape == 0), wide_candidates=(shape != 1),
        extra_refinement=(shape == -1), alternate_values=alternate,
    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=293)
    parser.add_argument("--datasets-dir", type=Path, default=Path("local_fitted"))
    parser.add_argument("--groups", default="0,1,2,3,4,5,6,9")
    parser.add_argument("--tests", default="0,1,2,3,4")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.0625)
    args = parser.parse_args()
    selected_groups = {int(x) for x in args.groups.split(",") if x}
    selected_tests = {int(x) for x in args.tests.split(",") if x}
    solution = load_solution(args.version)
    groups = torch.load(args.datasets_dir / "linear.pt", map_location="cpu", weights_only=False)
    total_current = total_rotated = total_recycled = 0.0
    started = time.perf_counter()
    for gi, group in enumerate(groups):
        if gi not in selected_groups:
            continue
        weight = dequantize_nvfp4(*group["weight"]).float()
        if int(weight.shape[0]) < int(weight.shape[1]):
            continue
        baseline_weight = dequantize_hif4(quantize_to_hif4(weight), tuple(weight.shape)).float()
        calibrated = solution.hif4_calibration_and_quantize_weight(
            *group["weight"], group["calib_activation_list"]
        )
        parent_state = calibrated["activation_state"]
        target_weight = parent_state["unrestricted_target_weight"].float()
        shape = int(parent_state["linear_shape"])
        calibration = torch.cat([
            transform(solution, dequantize_nvfp4(*packed).float(), parent_state)
            for packed in group["calib_activation_list"]
        ])
        _, _, right = torch.linalg.svd(target_weight, full_matrices=False)
        low = right[-args.rank:].T.contiguous()
        rotation = minimal_subspace_rotation(low)
        rotated_weight = target_weight @ rotation
        rotated_activation = calibration @ rotation
        blocks = int(rotated_weight.shape[1]) // 64
        sensitivity = (
            rotated_activation.reshape(-1, blocks, 64).square().mean(0)
            * rotated_weight.reshape(-1, blocks, 64).square().sum(0)
        ).sum(-1)
        channel_sensitivity = (
            rotated_activation.square().mean(0)
            * rotated_weight.square().sum(0)
        )
        high_channels = channel_sensitivity[:-args.rank].argsort(
            descending=True
        )[:args.rank]
        recycled_weight = rotated_weight.clone()
        recycled_weight[:, -args.rank:] = (
            args.alpha * recycled_weight.index_select(1, high_channels)
        )
        variants = {}
        for name, candidate in (("rotated", rotated_weight), ("recycled", recycled_weight)):
            params = solution._quantize_linear_values(
                candidate, offsets=solution._LINEAR_WEIGHT_SCALE_OFFSETS
            )
            variants[name] = (
                solution._dequantize_linear_params(params).reshape_as(candidate).float(),
                install(solution, target_weight, params),
            )
        tail_fraction = float(
            torch.linalg.svdvals(target_weight)[-args.rank:].square().sum()
            / target_weight.square().sum()
        )
        print(
            f"g={gi} {group.get('profile')} tail={tail_fraction:.6g} "
            f"high0={int(high_channels[0])}",
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
            values = transform(solution, raw, parent_state)
            target = values @ target_weight.T
            center = values @ rotation
            scores = {}
            for name, (candidate_weight, state) in variants.items():
                params = exact_search(
                    solution, center, values if name == "recycled" else None,
                    target, state, shape,
                )
                qa = solution._dequantize_linear_params(params).reshape_as(values).float()
                scores[name] = (baseline_mse - float((target - qa @ candidate_weight.T).square().mean())) / baseline_mse
            total_current += current
            total_rotated += scores["rotated"]
            total_recycled += scores["recycled"]
            print(
                f"  t={ti} current={current:.6f} rotated={scores['rotated']:.6f} "
                f"recycled={scores['recycled']:.6f}", flush=True,
            )
    print(
        f"summary seconds={time.perf_counter() - started:.2f} current={total_current:.9f} "
        f"rotated={total_rotated:.9f} recycled={total_recycled:.9f}", flush=True,
    )


if __name__ == "__main__":
    main()
