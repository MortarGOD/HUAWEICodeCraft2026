#!/usr/bin/env python3
"""Scan full-width polar strength on selected fitted Linear groups."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_local import evaluate_linear


def load_solution(version: int):
    path = ROOT / "solution" / f"solution_v{version}.py"
    spec = importlib.util.spec_from_file_location(f"polar_v{version}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=288)
    parser.add_argument("--datasets-dir", type=Path, default=Path("local_fitted"))
    parser.add_argument("--groups", default="5,6")
    parser.add_argument("--alphas", default="0.25,0.5,0.75,1.0")
    args = parser.parse_args()
    indices = [int(item) for item in args.groups.split(",") if item]
    groups = torch.load(
        args.datasets_dir / "linear.pt", map_location="cpu", weights_only=False
    )
    groups = [groups[index] for index in indices]
    for alpha in [float(item) for item in args.alphas.split(",") if item]:
        solution = load_solution(args.version)
        solution._LINEAR_WEIGHT_POLAR_ALPHA = alpha
        started = time.perf_counter()
        scores, _ = evaluate_linear(solution, groups)
        print(
            f"alpha={alpha:.4f} sum={sum(scores):.6f} "
            f"seconds={time.perf_counter() - started:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
