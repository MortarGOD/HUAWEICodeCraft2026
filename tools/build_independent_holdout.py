#!/usr/bin/env python3
"""Build a calibration-only cross-group holdout donor manifest.

The provided 0818 tensors copy calibration values into the corresponding test
slots.  This tool does not synthesize quantized values or duplicate the 5.7 GB
source files.  It pairs structurally identical groups using distribution
features extracted only from calibration tensors, then emits a small JSON
manifest to stdout.  ``local_eval.py`` uses each paired group's tensors as an
independent test donor.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = ROOT / "datasets_0818"
FEATURE_SAMPLE_LIMIT = 65536


def _sample_flat(value: torch.Tensor) -> torch.Tensor:
    flat = value.detach().to(torch.float32).flatten()
    if flat.numel() <= FEATURE_SAMPLE_LIMIT:
        return flat
    step = max(1, flat.numel() // FEATURE_SAMPLE_LIMIT)
    return flat[::step][:FEATURE_SAMPLE_LIMIT]


def _tensor_features(value: torch.Tensor, is_scale: bool) -> list[float]:
    sampled = _sample_flat(value)
    if is_scale:
        sampled = torch.log2(sampled.abs().clamp_min(2.0 ** -48))
    absolute = sampled.abs()
    return [
        float(sampled.mean().item()),
        float(sampled.std(unbiased=False).item()),
        float(absolute.mean().item()),
        float(torch.sqrt(sampled.square().mean()).item()),
        float(absolute.amax().item()),
        float((sampled == 0).to(torch.float32).mean().item()),
    ]


def _pair_features(pair: tuple[torch.Tensor, torch.Tensor]) -> list[float]:
    return _tensor_features(pair[0], False) + _tensor_features(pair[1], True)


def _linear_signature(group: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(group["weight"][0].shape),
        tuple(tuple(item[0].shape) for item in group["calib_activation_list"]),
    )


def _attention_signature(group: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(group["q_num_heads"]),
        int(group["kv_num_heads"]),
        int(group["head_dim"]),
        tuple(tuple(item["q"][0].shape) for item in group["calib"]),
    )


def _linear_features(group: dict[str, Any]) -> list[float]:
    result: list[float] = []
    for pair in group["calib_activation_list"]:
        result.extend(_pair_features(pair))
    return result


def _attention_features(group: dict[str, Any]) -> list[float]:
    result: list[float] = []
    for item in group["calib"]:
        for role in ("q", "k", "v"):
            result.extend(_pair_features(item[role]))
    return result


def _standardize(features: torch.Tensor) -> torch.Tensor:
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-6)
    return (features - mean) / std


def _pair_nearest(indices: list[int], features: list[list[float]]) -> dict[int, int]:
    """Greedily pair nearest groups; use a three-cycle for an odd remainder."""
    matrix = _standardize(torch.tensor([features[i] for i in indices]))
    distances = torch.cdist(matrix, matrix).square()
    remaining = list(range(len(indices)))
    donors: dict[int, int] = {}

    while len(remaining) > 3:
        best: tuple[float, int, int] | None = None
        for pos, left in enumerate(remaining):
            for right in remaining[pos + 1:]:
                candidate = (float(distances[left, right].item()), left, right)
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        _, left, right = best
        donors[indices[left]] = indices[right]
        donors[indices[right]] = indices[left]
        remaining.remove(left)
        remaining.remove(right)

    if len(remaining) == 3:
        a, b, c = remaining
        # Pick the lowest-cost directed cycle; its reverse has the same distance
        # sum, so numeric index order provides deterministic tie-breaking.
        donors[indices[a]] = indices[b]
        donors[indices[b]] = indices[c]
        donors[indices[c]] = indices[a]
    elif len(remaining) == 2:
        a, b = remaining
        donors[indices[a]] = indices[b]
        donors[indices[b]] = indices[a]
    elif remaining:
        raise ValueError("a structural signature has only one group")
    return donors


def _build_mapping(
    groups: list[dict[str, Any]],
    signature_fn,
    feature_fn,
) -> dict[str, int]:
    signatures: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    features: list[list[float]] = []
    for index, group in enumerate(groups):
        signatures[signature_fn(group)].append(index)
        features.append(feature_fn(group))

    mapping: dict[int, int] = {}
    for indices in signatures.values():
        mapping.update(_pair_nearest(indices, features))
    if len(mapping) != len(groups) or any(index == donor for index, donor in mapping.items()):
        raise RuntimeError("failed to construct a complete non-self donor mapping")
    return {str(index): int(mapping[index]) for index in range(len(groups))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    args = parser.parse_args()

    linear = torch.load(
        args.dataset_dir / "linear.pt", weights_only=True, map_location="cpu"
    )
    linear_mapping = _build_mapping(linear, _linear_signature, _linear_features)
    del linear

    attention = torch.load(
        args.dataset_dir / "attn.pt", weights_only=True, map_location="cpu"
    )
    attention_mapping = _build_mapping(
        attention, _attention_signature, _attention_features
    )

    manifest = {
        "schema_version": 1,
        "source": "datasets_0818",
        "method": "nearest calibration-feature donor within exact signature",
        "feature_sample_limit": FEATURE_SAMPLE_LIMIT,
        "linear_donors": linear_mapping,
        "attention_donors": attention_mapping,
    }
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
