#!/usr/bin/env python3
"""Fit a conservative online-score estimator from real-weight proxy results."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_local import evaluate_attention, evaluate_linear

# Only successful submissions on the current 50 Linear / 250 Attention set.
# Component values marked as estimates are diagnostics only; model selection is
# trained against exact total scores and never against these inferred splits.
ONLINE = {
    277: {"score": 17648.0, "seconds": 279.0, "linear_est": 9345.0, "attention_est": 8303.0},
    278: {"score": 17648.0, "seconds": 252.0, "linear_est": 9345.0, "attention_est": 8303.0},
    279: {"score": 17657.0, "seconds": 252.0, "linear_est": 9354.0, "attention_est": 8303.0},
    280: {"score": 17658.0, "seconds": 256.0, "linear_est": 9355.0, "attention_est": 8303.0},
    281: {"score": 17659.0, "seconds": 254.0, "linear_est": 9356.0, "attention_est": 8303.0},
    283: {"score": 17723.0, "seconds": 259.0, "linear_est": 9356.0, "attention_est": 8367.0},
    284: {"score": 17656.0, "seconds": 261.0, "linear_est": 9356.0, "attention_est": 8300.0},
    285: {"score": 17733.0, "seconds": 257.0, "linear_est": 9366.0, "attention_est": 8367.0},
    287: {"score": 17716.0, "seconds": 269.0, "linear_est": 9349.0, "attention_est": 8367.0},
    293: {"score": 17705.0, "seconds": 267.0, "linear_est": 9349.0, "attention_est": 8356.0},
    296: {"score": 17717.0, "seconds": 237.0, "linear_est": 9361.0, "attention_est": 8356.0},
    301: {"score": 17562.0, "seconds": 203.0, "linear_est": 9195.0, "attention_est": 8367.0},
    302: {"score": 19392.0, "seconds": 281.0, "linear_est": 9195.0, "attention_est": 10197.0},
}


def load_solution(version: int):
    path = ROOT / "solution" / f"solution_v{version}.py"
    spec = importlib.util.spec_from_file_location(f"estimator_v{version}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_sums(scores: list[float], groups: list[dict], field: str) -> dict[str, float]:
    result: dict[str, float] = {}
    cursor = 0
    for group in groups:
        count = len(group[field])
        source = group["source_model"]
        result[source] = result.get(source, 0.0) + sum(scores[cursor:cursor + count])
        cursor += count
    if cursor != len(scores):
        raise RuntimeError(f"score/group mismatch: {cursor} != {len(scores)}")
    return result


def evaluate_version(version: int, linear: list[dict], attention: list[dict]) -> dict:
    solution = load_solution(version)
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(20260902)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            linear_scores, linear_seconds = evaluate_linear(solution, linear)
            attention_scores, attention_seconds = evaluate_attention(solution, attention)
    finally:
        torch.random.set_rng_state(rng_state)
    linear_sources = _source_sums(linear_scores, linear, "test_activation_list")
    attention_sources = _source_sums(attention_scores, attention, "test")
    qwen = "Qwen/Qwen3.6-35B-A3B"
    deepseek = "deepseek-ai/DeepSeek-V4-Flash-0731"
    return {
        "linear": sum(linear_scores),
        "attention": sum(attention_scores),
        "total": sum(linear_scores) + sum(attention_scores),
        "linear_qwen": linear_sources[qwen],
        "linear_deepseek": linear_sources[deepseek],
        "attention_qwen": attention_sources[qwen],
        "attention_deepseek": attention_sources[deepseek],
        "linear_seconds": linear_seconds,
        "attention_seconds": attention_seconds,
        "local_seconds": linear_seconds + attention_seconds,
    }


def reevaluate_attention(version: int, attention: list[dict], previous: dict) -> dict:
    solution = load_solution(version)
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(20260902)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            scores, seconds = evaluate_attention(solution, attention)
    finally:
        torch.random.set_rng_state(rng_state)
    sources = _source_sums(scores, attention, "test")
    result = dict(previous)
    result.update({
        "attention": sum(scores),
        "attention_qwen": sources["Qwen/Qwen3.6-35B-A3B"],
        "attention_deepseek": sources["deepseek-ai/DeepSeek-V4-Flash-0731"],
        "attention_seconds": seconds,
        "total": result["linear"] + sum(scores),
        "local_seconds": result["linear_seconds"] + seconds,
    })
    return result


FEATURE_SETS = {
    "total": ("total",),
    "linear_attention": ("linear", "attention"),
    "model_components": (
        "linear_qwen", "linear_deepseek", "attention_qwen", "attention_deepseek"
    ),
    "core_stress": ("total", "stress_total"),
    "core_stress_components": (
        "linear", "attention", "stress_attention_qwen", "stress_attention_deepseek"
    ),
}


def _fit_ridge(x: torch.Tensor, y: torch.Tensor, penalty: float) -> dict:
    mean = x.mean(dim=0)
    scale = x.std(dim=0).clamp_min(1e-8)
    z = (x - mean) / scale
    design = torch.cat((torch.ones(z.shape[0], 1), z), dim=1)
    gram = design.T @ design
    regularizer = torch.eye(gram.shape[0]) * penalty
    regularizer[0, 0] = 0.0
    coefficients = torch.linalg.solve(gram + regularizer, design.T @ y)
    return {
        "mean": mean,
        "scale": scale,
        "coefficients": coefficients,
        "penalty": penalty,
    }


def _predict(model: dict, x: torch.Tensor) -> torch.Tensor:
    z = (x - model["mean"]) / model["scale"]
    design = torch.cat((torch.ones(z.shape[0], 1), z), dim=1)
    return design @ model["coefficients"]


def _loo_metrics(x: torch.Tensor, y: torch.Tensor, penalty: float) -> dict:
    predictions = []
    for holdout in range(y.numel()):
        keep = torch.arange(y.numel()) != holdout
        model = _fit_ridge(x[keep], y[keep], penalty)
        predictions.append(float(_predict(model, x[holdout:holdout + 1])[0]))
    prediction = torch.tensor(predictions, dtype=torch.float64)
    error = prediction - y
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(error.square().mean().sqrt()),
        "max_abs": float(error.abs().max()),
        "predictions": predictions,
    }


def fit_score_models(results: dict[int, dict]) -> dict:
    versions = sorted(results)
    target = torch.tensor([ONLINE[v]["score"] for v in versions], dtype=torch.float64)
    candidates = []
    for name, fields in FEATURE_SETS.items():
        if any(field not in results[versions[0]] for field in fields):
            continue
        x = torch.tensor(
            [[results[v][field] for field in fields] for v in versions],
            dtype=torch.float64,
        )
        for penalty in (0.01, 0.1, 1.0, 10.0, 100.0):
            metrics = _loo_metrics(x, target, penalty)
            candidates.append({
                "name": name,
                "fields": list(fields),
                "penalty": penalty,
                **metrics,
            })
    candidates.sort(key=lambda item: (item["mae"], item["rmse"]))
    winner = candidates[0]
    fields = tuple(winner["fields"])
    x = torch.tensor(
        [[results[v][field] for field in fields] for v in versions], dtype=torch.float64
    )
    fitted = _fit_ridge(x, target, winner["penalty"])
    fitted_prediction = _predict(fitted, x)
    return {
        "selected": winner,
        "training_predictions": {
            str(version): float(value)
            for version, value in zip(versions, fitted_prediction)
        },
        "model": {
            "fields": list(fields),
            "penalty": winner["penalty"],
            "mean": fitted["mean"].tolist(),
            "scale": fitted["scale"].tolist(),
            "coefficients": fitted["coefficients"].tolist(),
        },
        "candidates": candidates,
    }


def estimate(model_record: dict, local: dict) -> float:
    model = model_record["model"]
    x = torch.tensor(
        [[local[field] for field in model["fields"]]], dtype=torch.float64
    )
    fitted = {
        "mean": torch.tensor(model["mean"], dtype=torch.float64),
        "scale": torch.tensor(model["scale"], dtype=torch.float64),
        "coefficients": torch.tensor(model["coefficients"], dtype=torch.float64),
    }
    return float(_predict(fitted, x)[0])


def _correlation(left: list[float], right: list[float]) -> float:
    pair = torch.tensor([left, right], dtype=torch.float64)
    return float(torch.corrcoef(pair)[0, 1])


def build_diagnostics(results: dict[int, dict]) -> dict:
    versions = sorted(results)
    return {
        "total_pearson": _correlation(
            [results[v]["total"] for v in versions],
            [ONLINE[v]["score"] for v in versions],
        ),
        "linear_component_pearson": _correlation(
            [results[v]["linear"] for v in versions],
            [ONLINE[v]["linear_est"] for v in versions],
        ),
        "attention_component_pearson": _correlation(
            [results[v]["attention"] for v in versions],
            [ONLINE[v]["attention_est"] for v in versions],
        ),
        "training_local_total_range": [
            min(results[v]["total"] for v in versions),
            max(results[v]["total"] for v in versions),
        ],
        "policy": {
            "linear": (
                "use parent-anchored deltas and the empirical interval; local gains "
                "are direction evidence, not one-to-one score gains"
            ),
            "attention": (
                "current proxy is not score-predictive; require non-regression but "
                "estimate score from nearest online ablation until real activations exist"
            ),
            "extrapolation": "disabled outside the training local-score range",
        },
    }


def build_v41_anchor(results: dict[int, dict]) -> dict:
    v285, v301, v302 = results[285], results[301], results[302]
    attention_local_delta = v302["stress_attention"] - v301["stress_attention"]
    linear_local_delta = v301["linear"] - v285["linear"]
    return {
        "base_version": 302,
        "base_online_score": ONLINE[302]["score"],
        "base_local_linear": v302["linear"],
        "base_stress_attention": v302["stress_attention"],
        "attention_online_per_stress_point": (
            (ONLINE[302]["score"] - ONLINE[301]["score"]) / attention_local_delta
        ),
        "linear_online_per_core_point": (
            (ONLINE[301]["linear_est"] - ONLINE[285]["linear_est"]) / linear_local_delta
        ),
        "minimum_uncertainty": 100.0,
        "warning": (
            "single v41-family edge; use only for children of v302 and keep at "
            "least +/-100 until another online v41-derived result is observed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets-dir", type=Path, default=Path("local_fitted_models_realw")
    )
    parser.add_argument(
        "--versions", default=",".join(str(version) for version in ONLINE)
    )
    parser.add_argument("--predict-version", type=int)
    parser.add_argument(
        "--prediction-family", choices=("auto", "legacy", "v41", "global"),
        default="auto",
        help=(
            "prediction calibration family; auto uses legacy below v300 and the "
            "v41 parent anchor from v300 onward"
        ),
    )
    parser.add_argument(
        "--json", type=Path,
        default=Path("local_fitted_models_realw/online-score-model.json"),
    )
    parser.add_argument(
        "--stress-json", type=Path,
        default=Path("local_fitted_models_realw_v41/online-score-model.json"),
        help="optional fitted-context bank results used as additional features",
    )
    parser.add_argument(
        "--reuse-training", action="store_true",
        help="load the existing JSON fit and only evaluate --predict-version",
    )
    parser.add_argument(
        "--reuse-evaluations", action="store_true",
        help="reuse per-version local results in the JSON and evaluate only missing versions",
    )
    parser.add_argument(
        "--reevaluate-attention", action="store_true",
        help="reuse cached Linear results but recompute Attention for every version",
    )
    args = parser.parse_args()
    if args.predict_version is not None and args.predict_version >= 303:
        parser.error(
            "The old v41 stress anchor is invalidated by v303-v307 online feedback. "
            "Use the covariance bank and replay_major_edges.py; do not extrapolate "
            "a precise online score from the obsolete single-edge fit."
        )
    versions = [int(item) for item in args.versions.split(",") if item]
    unknown = [version for version in versions if version not in ONLINE]
    if unknown:
        parser.error(f"training versions lack online labels: {unknown}")

    linear = torch.load(
        args.datasets_dir / "linear.pt", map_location="cpu", weights_only=False
    )
    attention = torch.load(
        args.datasets_dir / "attn.pt", map_location="cpu", weights_only=False
    )
    if args.reuse_training:
        if not args.json.exists():
            parser.error(f"cannot reuse missing model: {args.json}")
        report = json.loads(args.json.read_text(encoding="utf-8"))
        results = {int(k): value for k, value in report["local"].items()}
        score_model = report["score_model"]
        report["diagnostics"] = report.get("diagnostics") or build_diagnostics(results)
    else:
        results: dict[int, dict] = {}
        if args.reuse_evaluations and args.json.exists():
            previous = json.loads(args.json.read_text(encoding="utf-8"))
            if previous.get("dataset") == str(args.datasets_dir):
                results.update({
                    int(k): value for k, value in previous.get("local", {}).items()
                    if int(k) in versions
                })
        for version in versions:
            if args.reevaluate_attention and version in results:
                started = time.perf_counter()
                results[version] = reevaluate_attention(
                    version, attention, results[version]
                )
                results[version]["wall_seconds"] = time.perf_counter() - started
                local = results[version]
                print(
                    f"v{version}: refreshed A={local['attention']:.6f} "
                    f"total={local['total']:.6f} time={local['wall_seconds']:.2f}s",
                    flush=True,
                )
                continue
            if version in results:
                print(f"v{version}: reused local evaluation", flush=True)
                continue
            started = time.perf_counter()
            local = evaluate_version(version, linear, attention)
            local["wall_seconds"] = time.perf_counter() - started
            results[version] = local
            print(
                f"v{version}: local={local['total']:.6f} "
                f"L={local['linear']:.6f} A={local['attention']:.6f} "
                f"online={ONLINE[version]['score']:.0f} "
                f"time={local['wall_seconds']:.2f}s",
                flush=True,
            )
        if args.stress_json.exists():
            stress_report = json.loads(args.stress_json.read_text(encoding="utf-8"))
            stress_local = stress_report.get("local", {})
            for version, local in results.items():
                other = stress_local.get(str(version))
                if other is None:
                    continue
                for field, value in other.items():
                    if field in {"linear", "attention", "total", "linear_qwen",
                                 "linear_deepseek", "attention_qwen",
                                 "attention_deepseek"}:
                        local[f"stress_{field}"] = value
        score_model = fit_score_models(results)
        legacy_results = {version: value for version, value in results.items() if version < 300}
        legacy_score_model = fit_score_models(legacy_results)
        v41_anchor = build_v41_anchor(results)
    selected = score_model["selected"]
    print(
        f"selected={selected['name']} ridge={selected['penalty']:g} "
        f"LOO_MAE={selected['mae']:.2f} LOO_RMSE={selected['rmse']:.2f} "
        f"LOO_MAX={selected['max_abs']:.2f}"
    )
    if not args.reuse_training:
        report = {
            "dataset": str(args.datasets_dir),
            "stress_dataset": str(args.stress_json.parent) if args.stress_json.exists() else None,
            "dataset_policy": "checkpoint weights only; legacy synthetic sets archived",
            "online_case_mix": {"linear": 50, "attention": 250},
            "versions": versions,
            "online": ONLINE,
            "local": results,
            "score_model": score_model,
            "legacy_score_model": legacy_score_model,
            "v41_anchor": v41_anchor,
            "diagnostics": build_diagnostics(results),
        }
    if args.predict_version is not None:
        local = evaluate_version(args.predict_version, linear, attention)
        family = args.prediction_family
        if family == "auto":
            family = "v41" if args.predict_version >= 300 else "legacy"
        if family in {"v41", "global"}:
            stress_dir = args.stress_json.parent
            stress_linear = torch.load(
                stress_dir / "linear.pt", map_location="cpu", weights_only=False
            )
            stress_attention = torch.load(
                stress_dir / "attn.pt", map_location="cpu", weights_only=False
            )
            stress = evaluate_version(
                args.predict_version, stress_linear, stress_attention
            )
            for field, value in stress.items():
                if field in {"linear", "attention", "total", "linear_qwen",
                             "linear_deepseek", "attention_qwen",
                             "attention_deepseek"}:
                    local[f"stress_{field}"] = value
        if family == "legacy":
            active_model = report["legacy_score_model"]
            prediction = estimate(active_model, local)
            active_selected = active_model["selected"]
            uncertainty = max(
                active_selected["max_abs"], 2.0 * active_selected["rmse"]
            )
            legacy_versions = [v for v in results if v < 300]
            low = min(results[v]["total"] for v in legacy_versions)
            high = max(results[v]["total"] for v in legacy_versions)
            extrapolated = not (low <= local["total"] <= high)
        elif family == "v41":
            anchor = report["v41_anchor"]
            prediction = (
                anchor["base_online_score"]
                + anchor["linear_online_per_core_point"]
                * (local["linear"] - anchor["base_local_linear"])
                + anchor["attention_online_per_stress_point"]
                * (local["stress_attention"] - anchor["base_stress_attention"])
            )
            uncertainty = anchor["minimum_uncertainty"]
            extrapolated = False
        else:
            prediction = estimate(score_model, local)
            uncertainty = max(selected["max_abs"], 2.0 * selected["rmse"])
            low, high = report["diagnostics"]["training_local_total_range"]
            extrapolated = not (low <= local["total"] <= high)
        report["prediction"] = {
            "version": args.predict_version,
            "family": family,
            "local": local,
            "online_score_center": prediction,
            "empirical_interval": [prediction - uncertainty, prediction + uncertainty],
            "extrapolated": extrapolated,
        }
        print(
            f"prediction v{args.predict_version}: {prediction:.1f} "
            f"interval≈[{prediction - uncertainty:.1f}, "
            f"{prediction + uncertainty:.1f}] extrapolated={extrapolated}"
        )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
