#!/usr/bin/env python3
"""Probe multi-threshold Q/K candidate banks under the existing surrogate."""

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


def install_bank(solution, thresholds: tuple[float, ...]) -> None:
    def quantize(values, offsets, energy_penalty=0.0):
        candidates = [
            solution._quantize_linear_values(
                values, offsets=(offset,), rounding_threshold=threshold
            )
            for threshold in thresholds
            for offset in offsets
        ]
        channels = int(values.shape[-1])
        target = values.float().reshape(
            *values.shape[:-1], channels // 64, 64
        )
        target_energy = target.square().sum(-1).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        scores = []
        for params in candidates:
            reconstructed = solution._dequantize_linear_params(params).reshape_as(target)
            residual = (target - reconstructed).square().sum(-1)
            if energy_penalty:
                drift = target_energy - reconstructed.square().sum(-1)
                residual = residual + energy_penalty * drift.square() / target_energy
            scores.append(residual)
        best = torch.stack(scores).argmin(0)
        result = {}
        for name in candidates[0]:
            stacked = torch.stack([candidate[name] for candidate in candidates])
            tail = stacked.shape[1 + best.ndim:]
            index = best.reshape((1,) + best.shape + (1,) * len(tail))
            result[name] = torch.gather(
                stacked, 0, index.expand((1,) + best.shape + tail)
            ).squeeze(0)
        return result
    solution._quantize_attention_values = quantize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--version", type=int, default=291)
    args = parser.parse_args()
    datasets = {
        name: torch.load(Path(name) / "attn.pt", weights_only=False)
        for name in args.datasets
    }
    banks = (
        (0.5,),
        (0.4375, 0.5, 0.5625),
        (0.375, 0.5, 0.625),
        (0.45, 0.5, 0.55),
    )
    for bank in banks:
        solution = load_solution(Path(f"solution/solution_v{args.version}.py").resolve())
        install_bank(solution, bank)
        for name, groups in datasets.items():
            with contextlib.redirect_stdout(io.StringIO()):
                scores, elapsed = evaluate_attention(solution, groups)
            print(bank, name, f"score={sum(scores):.6f}", f"seconds={elapsed:.2f}")


if __name__ == "__main__":
    main()
