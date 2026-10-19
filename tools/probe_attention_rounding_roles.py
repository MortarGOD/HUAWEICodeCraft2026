#!/usr/bin/env python3
"""Isolate Q/K/V contributions of a multi-rounding candidate bank."""

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
from probe_attention_rounding import install_bank


def install_role_bank(solution, roles: str, thresholds: tuple[float, ...]) -> None:
    original_helper = solution._quantize_attention_values
    install_bank(solution, thresholds)
    bank_helper = solution._quantize_attention_values
    solution._quantize_attention_values = original_helper
    for role in roles:
        name = f"hif4_dynamic_quantize_{role}"
        original = getattr(solution, name)

        def wrapped(*args, _original=original, **kwargs):
            saved = solution._quantize_attention_values
            solution._quantize_attention_values = bank_helper
            try:
                return _original(*args, **kwargs)
            finally:
                solution._quantize_attention_values = saved

        setattr(solution, name, wrapped)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--version", type=int, default=291)
    parser.add_argument("--thresholds", default="0.45,0.5,0.55")
    args = parser.parse_args()
    thresholds = tuple(float(x) for x in args.thresholds.split(",") if x)
    datasets = {
        name: torch.load(Path(name) / "attn.pt", weights_only=False)
        for name in args.datasets
    }
    for roles in ("q", "k", "v", "qk", "qv", "kv", "qkv"):
        solution = load_solution(Path(f"solution/solution_v{args.version}.py").resolve())
        install_role_bank(solution, roles, thresholds)
        for name, groups in datasets.items():
            with contextlib.redirect_stdout(io.StringIO()):
                scores, elapsed = evaluate_attention(solution, groups)
            print(roles, name, f"score={sum(scores):.6f}", f"seconds={elapsed:.2f}")


if __name__ == "__main__":
    main()
