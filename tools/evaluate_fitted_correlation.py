#!/usr/bin/env python3
"""Rank solution versions on the fitted dataset against known online deltas."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hif4_quant import dequantize_hif4, dequantize_nvfp4, quantize_to_hif4


# Linear deltas are identifiable because v10 Attention is frozen in v11-v30.
ONLINE_LINEAR_DELTA = {
    10: 0.0,
    11: 1810.0,
    12: 1721.0,
    13: 2004.0,
    14: 1839.0,
    15: 2349.0,
    16: 2491.0,
    17: 2506.0,
    18: 2478.0,
    19: 2455.0,
    20: 3266.0,
    21: 2688.0,
    22: 3261.0,
    23: 3261.0,
    24: 3318.0,
    25: 3330.0,
    26: 2982.0,
    27: 3054.0,
    28: 3346.0,
    29: 7234.0,
    30: 6399.0,
}

# v32+ kept the same Attention path in the submitted ablations, so raw total
# score is a valid target for their relative Linear ordering.  These points
# directly supervise the fitted dataset's cross-block covariance axis.
ONLINE_TOTAL_SCORE = {
    32: 18945.0,
    33: 18944.0,
    36: 18915.0,
    37: 18882.0,
    39: 18807.0,
    40: 18514.0,
    42: 18051.0,
    73: 18146.0,
    113: 18330.0,
    117: 18949.3,
    118: 18949.3,
    119: 18949.5,
    120: 18949.5,
}


def load_solution(version: int):
    path = ROOT / "solution" / f"solution_v{version}.py"
    name = f"fitted_solution_v{version}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def output_mses(
    activation: torch.Tensor,
    weight: torch.Tensor,
    baseline_a: torch.Tensor,
    baseline_w: torch.Tensor,
    player_a: torch.Tensor,
    player_w: torch.Tensor,
    chunk: int,
) -> tuple[float, float]:
    baseline_error = 0.0
    player_error = 0.0
    elements = 0
    for start in range(0, weight.shape[0], chunk):
        stop = min(start + chunk, weight.shape[0])
        reference = activation @ weight[start:stop].T
        baseline = baseline_a @ baseline_w[start:stop].T
        player = player_a @ player_w[start:stop].T
        baseline_error += float((reference - baseline).square().sum())
        player_error += float((reference - player).square().sum())
        elements += reference.numel()
    return baseline_error / elements, player_error / elements


@torch.inference_mode()
def evaluate_linear(version: int, groups: list[dict], output_chunk: int) -> tuple[float, dict[str, float]]:
    solution = load_solution(version)
    total = 0.0
    profile_scores: dict[str, list[float]] = {}
    for group in groups:
        wq, ws = group["weight"]
        weight = dequantize_nvfp4(wq, ws).float()
        baseline_w = dequantize_hif4(quantize_to_hif4(weight), tuple(weight.shape)).float()
        result = solution.hif4_calibration_and_quantize_weight(
            wq, ws, group["calib_activation_list"]
        )
        player_w = dequantize_hif4(result["weight_params"], tuple(weight.shape)).float()
        for aq, scale in group["test_activation_list"]:
            activation = dequantize_nvfp4(aq, scale).float()
            baseline_a = dequantize_hif4(
                quantize_to_hif4(activation), tuple(activation.shape)
            ).float()
            player_params = solution.hif4_dynamic_quantize_activation(
                aq, scale, result["activation_state"]
            )
            player_a = dequantize_hif4(player_params, tuple(activation.shape)).float()
            baseline_mse, player_mse = output_mses(
                activation, weight, baseline_a, baseline_w,
                player_a, player_w, output_chunk,
            )
            case_score = (baseline_mse - player_mse) / max(baseline_mse, 1e-30)
            total += case_score
            profile_scores.setdefault(group.get("profile", "unknown"), []).append(case_score)
    means = {name: sum(scores) / len(scores) for name, scores in profile_scores.items()}
    return total, means


def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denom = left.square().sum().sqrt() * right.square().sum().sqrt()
    return float((left * right).sum() / denom.clamp_min(1e-30))


def ranks(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort()
    result = torch.empty_like(values)
    result[order] = torch.arange(values.numel(), dtype=values.dtype)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-dir", type=Path, default=Path("local_fitted"))
    parser.add_argument(
        "--versions", default="10,11,13,17,20,21,24,25,26,28,29,30"
    )
    parser.add_argument("--max-groups", type=int, default=50)
    parser.add_argument("--output-chunk", type=int, default=512)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    versions = [int(item) for item in args.versions.split(",") if item]
    if all(version in ONLINE_LINEAR_DELTA for version in versions):
        online_scores = ONLINE_LINEAR_DELTA
        online_label = "online_delta"
    elif all(version in ONLINE_TOTAL_SCORE for version in versions):
        online_scores = ONLINE_TOTAL_SCORE
        online_label = "online_total"
    else:
        known = set(ONLINE_LINEAR_DELTA) | set(ONLINE_TOTAL_SCORE)
        unknown = [version for version in versions if version not in known]
        raise ValueError(
            "versions must all belong to one comparable online series; "
            f"unknown={unknown}"
        )
    groups = torch.load(
        args.datasets_dir / "linear.pt", map_location="cpu", weights_only=False
    )[:args.max_groups]
    results: dict[int, dict] = {}
    for version in versions:
        started = time.perf_counter()
        score, profiles = evaluate_linear(version, groups, args.output_chunk)
        results[version] = {
            "local_score": score,
            online_label: online_scores[version],
            "seconds": time.perf_counter() - started,
            "profile_means": profiles,
        }
        print(
            f"v{version}: local={score:.6f} {online_label}={online_scores[version]:.0f} "
            f"seconds={results[version]['seconds']:.2f}",
            flush=True,
        )
    local = torch.tensor([results[v]["local_score"] for v in versions])
    online = torch.tensor([online_scores[v] for v in versions])
    pearson = correlation(local, online)
    spearman = correlation(ranks(local), ranks(online))
    report = {
        "versions": versions,
        "groups": len(groups),
        "online_metric": online_label,
        "pearson": pearson,
        "spearman": spearman,
        "results": results,
    }
    print(f"correlation: pearson={pearson:.6f} spearman={spearman:.6f}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
