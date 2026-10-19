#!/usr/bin/env python3
"""Probe an extra reciprocal Q/K scale around the calibrated balance."""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_local import evaluate_attention, load_solution


def install(solution, alpha: float) -> None:
    original = solution.hif4_calibration_attention

    def calibration(calib, q_heads, kv_heads, dim):
        states = original(calib, q_heads, kv_heads, dim)
        q_state, k_state = states["q_state"], states["k_state"]
        q_state["coefficient"] = q_state["coefficient"] * alpha
        k_state["coefficient"] = k_state["coefficient"] / alpha
        k_state["center"] = k_state["center"] / alpha
        if bool(q_state["predictor_enabled"]):
            q_state["predictor_center"] = q_state["predictor_center"] * alpha
            q_state["predictor_map"] = q_state["predictor_map"] / (alpha * alpha)
        if bool(k_state["predictor_enabled"]):
            k_state["predictor_map"] = k_state["predictor_map"] * (alpha * alpha)
        return states

    solution.hif4_calibration_attention = calibration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--version", type=int, default=296)
    parser.add_argument(
        "--alphas", nargs="+", type=float,
        default=(0.7, 0.85, 1.0, 1.175, 1.4),
    )
    args = parser.parse_args()
    for dataset in args.datasets:
        groups = torch.load(
            Path(dataset) / "attn.pt", map_location="cpu", weights_only=False
        )
        for alpha in args.alphas:
            solution = load_solution(
                ROOT / "solution" / f"solution_v{args.version}.py"
            )
            install(solution, alpha)
            with contextlib.redirect_stdout(io.StringIO()):
                scores, elapsed = evaluate_attention(solution, groups)
            print(
                dataset, f"alpha={alpha:g}", f"score={sum(scores):.6f}",
                f"seconds={elapsed:.2f}", flush=True,
            )


if __name__ == "__main__":
    main()
