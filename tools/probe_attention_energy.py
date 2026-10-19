#!/usr/bin/env python3
"""Scan the Q/K block-energy drift penalty across fitted datasets."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_local import evaluate_attention


def load_solution(version: int):
    path = ROOT / "solution" / f"solution_v{version}.py"
    spec = importlib.util.spec_from_file_location(f"energy_v{version}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--version", type=int, default=291)
    parser.add_argument("--penalties", default="0,0.03125,0.0625,0.125,0.25,0.5")
    args = parser.parse_args()
    loaded = {
        dataset: torch.load(
            Path(dataset) / "attn.pt", map_location="cpu", weights_only=False
        )
        for dataset in args.datasets
    }
    for penalty in [float(x) for x in args.penalties.split(",") if x]:
        solution = load_solution(args.version)
        solution._ATTENTION_ENERGY_PENALTY = penalty
        for dataset, groups in loaded.items():
            started = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                scores, _ = evaluate_attention(solution, groups)
            print(
                f"penalty={penalty:.6f} dataset={dataset} "
                f"score={sum(scores):.6f} seconds={time.perf_counter()-started:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
