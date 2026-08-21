"""Build an auditable layout-to-parameter profile bank on synthetic NVFP4 data.

This is a development-only tool.  Submission code must not read its output.
Q(A) profiles are ranked with activation reconstruction error only.  A @ W is
used solely to rank fixed Q(W) ridge candidates, matching the 0819-V2 rules.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
PROXY_SHAPES = (
    (256, 64),   # observed 8192 x 2048 aspect ratio
    (256, 128),  # observed 8192 x 4096 aspect ratio
    (128, 128),  # observed 4096 x 4096 aspect ratio
    (128, 256),  # observed 4096 x 8192 aspect ratio
)
ROW_SHAPES = (1, 10, 32, 64, 128)
LAYOUTS = (
    "gaussian",
    "laplace",
    "heavy_tail",
    "sparse_outlier",
    "channel_outlier",
    "block_outlier",
    "correlated",
    "heteroscedastic",
)
ACTIVATION_PROFILES = {
    "compact": (0, 1, 2),
    "low_range": (0, 1),
    "wide": (0, 1, 2, 3),
    "shifted": (1, 2, 3),
}
ACTIVATION_COMPLEXITY_PENALTY = {
    "compact": 0.0,
    "low_range": -0.001,
    "wide": 0.005,
    "shifted": 0.002,
}
RIDGE_PROFILES = ("baseline", "lambda10", "lambda20", "lambda40")
NVFP4_LEVELS = torch.tensor(
    (-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
      0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
    dtype=torch.float32,
)


def load_solution(path: Path):
    spec = importlib.util.spec_from_file_location("layout_search_solution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import solution from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def nvfp4_round(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Round a float tensor to the public NVFP4 carrier/block-scale layout."""
    groups = values.to(torch.float32).reshape(*values.shape[:-1], -1, 16)
    scale = (groups.abs().amax(-1) / 6.0).clamp_min(2.0 ** -24)
    normalized = groups / scale.unsqueeze(-1)
    levels = NVFP4_LEVELS.to(normalized.device)
    indices = (normalized.unsqueeze(-1) - levels).abs().argmin(-1)
    carrier = levels[indices].reshape_as(values).to(torch.bfloat16)
    return carrier, scale.to(torch.bfloat16)


def dequant_nvfp4(pair: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    carrier, scale = pair
    return (
        carrier.to(torch.float32).reshape(*carrier.shape[:-1], -1, 16)
        * scale.to(torch.float32).unsqueeze(-1)
    ).flatten(-2)


def generate_layout(
    rows: int,
    channels: int,
    layout: str,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    value = torch.randn(rows, channels, generator=generator)
    if layout == "gaussian":
        pass
    elif layout == "laplace":
        uniform = torch.rand(rows, channels, generator=generator).clamp_min(1e-6)
        value = torch.sign(value) * (-torch.log(uniform)) / math.sqrt(2.0)
    elif layout == "heavy_tail":
        denominator = torch.stack([
            torch.randn(rows, channels, generator=generator).square()
            for _ in range(3)
        ]).sum(0).div(3.0).sqrt().clamp_min(0.1)
        value = value / denominator
    elif layout == "sparse_outlier":
        mask = torch.rand(rows, channels, generator=generator) < 0.005
        value = torch.where(mask, value * 16.0, value)
    elif layout == "channel_outlier":
        scale = torch.ones(channels)
        count = max(1, channels // 32)
        indices = torch.randperm(channels, generator=generator)[:count]
        scale[indices] = 8.0
        value = value * scale
    elif layout == "block_outlier":
        blocks = channels // 64
        scale = torch.ones(blocks)
        count = max(1, blocks // 8)
        indices = torch.randperm(blocks, generator=generator)[:count]
        scale[indices] = 8.0
        value = value.reshape(rows, blocks, 64) * scale.reshape(1, blocks, 1)
        value = value.reshape(rows, channels)
    elif layout == "correlated":
        rank = min(8, rows, channels)
        latent = torch.randn(rows, rank, generator=generator)
        loading = torch.randn(rank, channels, generator=generator)
        value = 0.75 * (latent @ loading) / math.sqrt(rank) + 0.25 * value
    elif layout == "heteroscedastic":
        log_scale = torch.linspace(-1.5, 1.5, channels)
        permutation = torch.randperm(channels, generator=generator)
        value = value * log_scale.exp()[permutation]
    else:
        raise ValueError(f"unknown layout {layout!r}")
    return dequant_nvfp4(nvfp4_round(value))


def layout_features(weight: torch.Tensor, activations: list[torch.Tensor]) -> dict[str, float]:
    activation = torch.cat(activations)

    def stats(value: torch.Tensor, prefix: str) -> dict[str, float]:
        value = value.to(torch.float32)
        rms = value.square().mean().sqrt().clamp_min(1e-12)
        channel_rms = value.square().mean(0).sqrt().clamp_min(1e-12)
        groups = value.reshape(-1, value.shape[-1] // 64, 64)
        block_rms = groups.square().mean((0, 2)).sqrt().clamp_min(1e-12)
        adjacent = (
            value[:, :-1] * value[:, 1:]
        ).mean().abs() / value.square().mean().clamp_min(1e-12)
        return {
            f"{prefix}_peak_rms": float(value.abs().amax() / rms),
            f"{prefix}_channel_imbalance": float(
                torch.quantile(channel_rms, 0.95) / torch.quantile(channel_rms, 0.50)
            ),
            f"{prefix}_block_imbalance": float(
                block_rms.max() / block_rms.median()
            ),
            f"{prefix}_zero_fraction": float((value == 0).float().mean()),
            f"{prefix}_adjacent_correlation": float(adjacent),
        }

    result = {
        "log2_input": math.log2(weight.shape[1]),
        "log2_output": math.log2(weight.shape[0]),
        "log2_aspect": math.log2(weight.shape[0] / weight.shape[1]),
    }
    result.update(stats(weight, "weight"))
    result.update(stats(activation, "activation"))
    return result


def dequant_hif4(module: Any, params: dict[str, torch.Tensor]) -> torch.Tensor:
    return module._dequantize_linear_params(params)


def relative_mse(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    return (reference - candidate).square().mean() / reference.square().mean().clamp_min(1e-12)


def choose_activation_profile(module: Any, activations: list[torch.Tensor]) -> tuple[str, dict[str, float]]:
    """Choose Q(A) using A reconstruction only; never uses A @ W."""
    scores: dict[str, float] = {}
    for name, offsets in ACTIVATION_PROFILES.items():
        errors = []
        for activation in activations:
            params = module._quantize_linear_values(activation, offsets=offsets)
            errors.append(relative_mse(activation, dequant_hif4(module, params)))
        scores[name] = float(torch.stack(errors).mean())
    trusted = scores["compact"]
    utility = {
        name: score / trusted + ACTIVATION_COMPLEXITY_PENALTY[name]
        for name, score in scores.items()
    }
    return min(utility, key=utility.get), scores


def fit_ridge_candidate(
    module: Any,
    activation: torch.Tensor,
    quantized_activation: torch.Tensor,
    weight: torch.Tensor,
    ridge_lambda: float,
) -> dict[str, torch.Tensor]:
    kernel = quantized_activation @ quantized_activation.T
    target = activation @ weight.T
    residual = target - quantized_activation @ weight.T
    ridge_scale = kernel.diag().mean()
    identity = torch.eye(kernel.shape[0], dtype=kernel.dtype)
    dual = torch.linalg.solve(kernel + ridge_lambda * ridge_scale * identity, residual)
    fitted = weight + (quantized_activation.T @ dual).T
    importance = activation.square().mean(0)
    return module._quantize_linear_values(fitted, importance)


def choose_weight_profile(
    module: Any,
    weight: torch.Tensor,
    calibration: list[torch.Tensor],
    holdout: list[torch.Tensor],
    activation_profile: str,
) -> tuple[str, dict[str, float]]:
    """Rank only fixed Q(W) candidates with held-out A @ W output error."""
    offsets = ACTIVATION_PROFILES[activation_profile]
    activation = torch.cat(calibration)
    quantized_activation = dequant_hif4(
        module, module._quantize_linear_values(activation, offsets=offsets)
    )
    baseline = module._quantize_linear_values(weight, activation.square().mean(0))
    candidates = {"baseline": baseline}
    for ridge_lambda in (10.0, 20.0, 40.0):
        candidates[f"lambda{int(ridge_lambda)}"] = fit_ridge_candidate(
            module, activation, quantized_activation, weight, ridge_lambda
        )

    trusted_weight = dequant_hif4(module, candidates["lambda20"])
    trusted_deviation = (trusted_weight - weight).square().mean()
    scores: dict[str, float] = {}
    for name, params in candidates.items():
        quantized_weight = dequant_hif4(module, params)
        deviation = (quantized_weight - weight).square().mean()
        if name not in ("baseline", "lambda20") and deviation > 1.5 * trusted_deviation:
            scores[name] = math.inf
            continue
        errors = []
        for activation_holdout in holdout:
            quantized_holdout = dequant_hif4(
                module,
                module._quantize_linear_values(
                    activation_holdout,
                    offsets=offsets,
                ),
            )
            target = activation_holdout @ weight.T
            output = quantized_holdout @ quantized_weight.T
            errors.append(relative_mse(target, output))
        scores[name] = float(torch.stack(errors).mean())
    return min(scores, key=scores.get), scores


def run_search(solution_path: Path, seeds: int) -> dict[str, Any]:
    module = load_solution(solution_path)
    records = []
    for layout_index, layout in enumerate(LAYOUTS):
        for shape_index, (output_dim, input_dim) in enumerate(PROXY_SHAPES):
            for seed in range(seeds):
                base_seed = 100000 * layout_index + 1000 * shape_index + seed
                weight = generate_layout(output_dim, input_dim, layout, base_seed)
                calibration = [
                    generate_layout(rows, input_dim, layout, base_seed + 10 + i)
                    for i, rows in enumerate(ROW_SHAPES)
                ]
                holdout = [
                    generate_layout(rows, input_dim, layout, base_seed + 100 + i)
                    for i, rows in enumerate(ROW_SHAPES)
                ]
                activation_profile, activation_scores = choose_activation_profile(
                    module, calibration
                )
                weight_profile, weight_scores = choose_weight_profile(
                    module,
                    weight,
                    calibration,
                    holdout,
                    activation_profile,
                )
                records.append({
                    "layout": layout,
                    "shape": [output_dim, input_dim],
                    "seed": seed,
                    "features": layout_features(weight, calibration),
                    "activation_profile": activation_profile,
                    "activation_scores": activation_scores,
                    "weight_profile": weight_profile,
                    "weight_scores": {
                        name: score if math.isfinite(score) else None
                        for name, score in weight_scores.items()
                    },
                })
                print(
                    f"layout={layout} shape={output_dim}x{input_dim} seed={seed} "
                    f"activation={activation_profile} weight={weight_profile}",
                    flush=True,
                )

    summary: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["layout"]].append(record)
    for layout, items in grouped.items():
        train = [item for item in items if item["seed"] < seeds - 1]
        heldout = [item for item in items if item["seed"] == seeds - 1]
        feature_names = list(items[0]["features"])
        centroid = {
            name: sum(item["features"][name] for item in train) / len(train)
            for name in feature_names
        }
        summary[layout] = {
            "feature_centroid": centroid,
            "train_activation_winners": dict(Counter(
                item["activation_profile"] for item in train
            )),
            "heldout_activation_winners": dict(Counter(
                item["activation_profile"] for item in heldout
            )),
            "train_weight_winners": dict(Counter(
                item["weight_profile"] for item in train
            )),
            "heldout_weight_winners": dict(Counter(
                item["weight_profile"] for item in heldout
            )),
        }
    return {
        "schema_version": 1,
        "compliance": {
            "activation_selection_uses_output": False,
            "weight_selection_may_use_a_matmul_w": True,
            "submission_file_io_required": False,
        },
        "proxy_shapes": [list(shape) for shape in PROXY_SHAPES],
        "row_shapes": list(ROW_SHAPES),
        "seed_count": seeds,
        "summary": summary,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solution",
        type=Path,
        default=ROOT / "solution" / "solution_v16.py",
    )
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seeds < 2:
        raise ValueError("--seeds must be at least 2")
    result = run_search(args.solution, args.seeds)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
        print(f"wrote {args.output}")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
