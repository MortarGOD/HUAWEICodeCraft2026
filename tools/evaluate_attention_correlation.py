#!/usr/bin/env python3
"""Check synthetic Attention ordering against identifiable online deltas."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import evaluate_attention


ONLINE_ATTENTION_DELTA = {1: 0.0, 6: 864.0, 7: 865.0, 8: 837.0, 9: 843.0, 10: 868.0}


def load_solution(version: int):
    path = ROOT / "solution" / f"solution_v{version}.py"
    spec = importlib.util.spec_from_file_location(f"attention_v{version}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ranks(values: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(values)
    result[values.argsort()] = torch.arange(values.numel(), dtype=values.dtype)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-dir", type=Path, default=Path("local_fitted"))
    parser.add_argument("--versions", default="1,6,7,8,9,10")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    versions = [int(value) for value in args.versions.split(",") if value]
    data = torch.load(args.datasets_dir / "attn.pt", map_location="cpu", weights_only=False)
    local_scores = []
    for version in versions:
        with contextlib.redirect_stdout(io.StringIO()):
            scores, _ = evaluate_attention(load_solution(version), data)
        local_scores.append(sum(scores))
        print(
            f"v{version}: local={local_scores[-1]:.6f} "
            f"online_delta={ONLINE_ATTENTION_DELTA[version]:.0f}"
        )
    local = torch.tensor(local_scores)
    online = torch.tensor([ONLINE_ATTENTION_DELTA[v] for v in versions])
    pearson = float(torch.corrcoef(torch.stack((local, online)))[0, 1])
    spearman = float(torch.corrcoef(torch.stack((ranks(local), ranks(online))))[0, 1])
    report = {
        "versions": versions,
        "local_scores": local_scores,
        "online_deltas": [ONLINE_ATTENTION_DELTA[v] for v in versions],
        "pearson": pearson,
        "spearman": spearman,
    }
    print(f"correlation: pearson={pearson:.6f} spearman={spearman:.6f}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
