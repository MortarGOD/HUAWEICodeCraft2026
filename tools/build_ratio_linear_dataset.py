#!/usr/bin/env python3
"""Build non-integer-width Linear down profiles for residual-frame tests."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_fitted_dataset as fitted


def build_group(
    profile: fitted.Profile,
    seed: int,
    output_channels: int,
    input_channels: int,
    rows: tuple[int, ...],
    ratio_name: str,
) -> dict:
    generator = torch.Generator().manual_seed(seed)
    weight, column_scale = fitted._raw_weight(
        output_channels, input_channels, profile, generator
    )
    activation_rank = min(profile.activation_rank, input_channels)
    shared_basis = torch.randn(
        activation_rank, input_channels, generator=generator
    )
    shared_basis /= shared_basis.square().mean(
        dim=1, keepdim=True
    ).sqrt().clamp_min(1.0e-6)
    shared_test_drift = torch.exp(
        torch.randn(input_channels, generator=generator) * profile.test_drift
    ).clamp(0.25, 4.0)
    group = {
        "key": f"linear/{profile.name}_{ratio_name}/{seed}",
        "profile": f"{profile.name}_{ratio_name}",
        "logical_strides": list(fitted.STRIDES),
        "weight": fitted.quantize_nvfp4(weight),
        "calib_activation_list": [],
        "test_activation_list": [],
    }
    for test in (False, True):
        destination = (
            group["test_activation_list"]
            if test else group["calib_activation_list"]
        )
        for count in rows:
            activation = fitted._raw_activation(
                count,
                input_channels,
                column_scale,
                profile,
                generator,
                test,
                shared_basis,
                shared_test_drift,
            )
            destination.append(fitted.quantize_nvfp4(activation))
    return group


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("local_fitted_ratio"))
    parser.add_argument("--base", type=int, default=512)
    parser.add_argument("--full-rows", action="store_true")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument(
        "--ratio-values", default="2.625,2.75,3.5",
        help="comma-separated input/output ratios",
    )
    parser.add_argument(
        "--profile-count", type=int, default=5,
        help="take this many profiles from the five-profile down bank",
    )
    args = parser.parse_args()
    if args.base % 64:
        raise ValueError("base must be divisible by 64")
    ratio_values = [
        float(value) for value in args.ratio_values.split(",") if value
    ]
    ratios = tuple(
        ("r" + format(value, "g").replace(".", "p"), value)
        for value in ratio_values
    )
    profile_bank = (
        fitted.PROFILES[7],
        fitted.PROFILES[8],
        replace(fitted.PROFILES[4], name="down_block", shape="down"),
        replace(fitted.PROFILES[9], name="down_sparse_shift", shape="down"),
        replace(
            fitted.PROFILES[0], name="down_balanced_shift",
            shape="down", test_drift=0.05,
        ),
    )
    profiles = profile_bank[:args.profile_count]
    rows = fitted.STRIDES if args.full_rows else (1, 10, 32, 64, 128)
    groups = []
    for ratio_index, (ratio_name, ratio) in enumerate(ratios):
        input_channels = int(round(args.base * ratio / 64.0)) * 64
        for profile_index, profile in enumerate(profiles):
            groups.append(build_group(
                profile,
                args.seed + 100 * ratio_index + profile_index,
                args.base,
                input_channels,
                rows,
                ratio_name,
            ))
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(groups, args.output / "linear.pt")
    manifest = {
        "seed": args.seed,
        "base": args.base,
        "rows": list(rows),
        "ratios": [name for name, _ in ratios],
        "groups": len(groups),
        "purpose": "v294 non-integer residual-frame validation",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
