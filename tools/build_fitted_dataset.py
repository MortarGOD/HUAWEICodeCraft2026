#!/usr/bin/env python3
"""Build a synthetic dataset supervised by the online ablation history.

The official mini_sample is deliberately not used as a distribution prior.  It
is useful only for the public container schema.  The generated profiles encode
properties implied by online results: channel imbalance, non-diagonal Weight
Gram structure, sparse block outliers, calibration/test drift, and mixed
square/up/down Linear shapes.

The default ``correlation`` preset follows the 2026-08-31 expected online mix:
10 Linear weights / 50 Linear test activations and 50 Attention groups / 250
Attention tests.  Dimensions are scaled down so many solution versions can be
ranked locally.  ``full`` uses 4K+ Linear dimensions and the real stride row
counts; it is intentionally several GB and expensive to score.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch


STRIDES = (1, 10, 128, 512, 1024)
# Seed 98 is a deliberately held-out zigzag-overfit case: v8 selects its extra
# calibration candidate but loses on the shifted tests, matching online v8<v7.
ATTENTION_SEED_OFFSETS = (14, 29, 42, 58, 60, 63, 73, 82, 89, 98)
NVFP4_CODEBOOK = torch.tensor(
    (-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
      0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
    dtype=torch.float32,
)


@dataclass(frozen=True)
class Profile:
    name: str
    shape: str
    channel_spread: float
    weight_rank: int
    correlated_strength: float
    tail_probability: float
    tail_multiplier: float
    block_spike: float
    test_drift: float
    sparsity: float = 0.0
    activation_rank: int = 8
    activation_strength: float = 0.65


PROFILES = (
    Profile("square_balanced", "square", 0.20, 0, 0.0, 0.001, 6.0, 0.0, 0.05,
            activation_strength=1.0),
    Profile("square_channel", "square", 1.00, 0, 0.0, 0.002, 8.0, 0.0, 0.12),
    Profile("square_gram", "square", 0.55, 16, 1.4, 0.001, 6.0, 0.0, 0.10),
    Profile("square_tail", "square", 0.65, 8, 0.8, 0.008, 12.0, 0.0, 0.18),
    Profile("square_block", "square", 0.50, 16, 1.0, 0.002, 8.0, 8.0, 0.15,
            activation_strength=1.0),
    Profile("up_gram", "up", 0.70, 16, 1.4, 0.002, 8.0, 0.0, 0.10),
    Profile("up_balanced", "up", 0.20, 0, 0.0, 0.001, 6.0, 0.0, 0.05,
            activation_strength=1.0),
    Profile("down_gram", "down", 0.70, 16, 1.5, 0.002, 8.0, 0.0, 0.12),
    Profile("down_balanced", "down", 0.20, 0, 0.0, 0.001, 6.0, 0.0, 0.05,
            activation_strength=1.0),
    Profile("square_sparse_shift", "square", 0.80, 12, 1.2, 0.004, 10.0, 3.0, 0.50, 0.12,
            activation_strength=1.0),
)

# v278 changed only square Linear and was exactly tied with v277 online.  Keep
# the original mixed bank for historical correlation, and provide this
# rectangular hypothesis bank for post-v278 development until an up-only probe
# distinguishes the online up/down mix.
RECTANGULAR_PROFILES = (
    PROFILES[5],
    PROFILES[6],
    replace(PROFILES[1], name="up_channel", shape="up"),
    replace(PROFILES[2], name="up_correlated", shape="up"),
    replace(PROFILES[3], name="up_tail", shape="up"),
    PROFILES[7],
    PROFILES[8],
    replace(PROFILES[4], name="down_block", shape="down"),
    replace(PROFILES[9], name="down_sparse_shift", shape="down"),
    replace(PROFILES[0], name="down_balanced_shift", shape="down", test_drift=0.10),
)


def _nearest_nvfp4(values: torch.Tensor) -> torch.Tensor:
    flat = values.reshape(-1, 1)
    indices = (flat - NVFP4_CODEBOOK.view(1, -1)).abs().argmin(dim=1)
    return NVFP4_CODEBOOK[indices].reshape_as(values)


def quantize_nvfp4(values: torch.Tensor, row_chunk: int = 256) -> list[torch.Tensor]:
    """Encode FP32 data into the public BF16 NVFP4 carrier/scale convention."""
    if values.ndim != 2 or values.shape[1] % 16:
        raise ValueError("NVFP4 input must be 2D with channels divisible by 16")
    quant_parts: list[torch.Tensor] = []
    scale_parts: list[torch.Tensor] = []
    for start in range(0, values.shape[0], row_chunk):
        part = values[start:start + row_chunk].float().reshape(
            -1, values.shape[1] // 16, 16
        )
        scale = (part.abs().amax(dim=-1) / 6.0).clamp_min(2.0 ** -9)
        scale = scale.clamp_max(448.0).to(torch.float8_e4m3fn).to(torch.float32)
        scale = scale.clamp_min(2.0 ** -9)
        carrier = _nearest_nvfp4((part / scale.unsqueeze(-1)).clamp(-6.0, 6.0))
        quant_parts.append(carrier.flatten(-2).to(torch.bfloat16))
        scale_parts.append(scale.to(torch.bfloat16))
    return [torch.cat(quant_parts), torch.cat(scale_parts)]


def _shape(profile: Profile, base: int) -> tuple[int, int]:
    if profile.shape == "square":
        return base, base
    if profile.shape == "up":
        return 2 * base, base
    if profile.shape == "down":
        return base, 2 * base
    raise ValueError(profile.shape)


def _channel_scales(channels: int, spread: float, generator: torch.Generator) -> torch.Tensor:
    scales = torch.exp(torch.randn(channels, generator=generator) * spread)
    return (scales / scales.log().mean().exp()).clamp(1.0 / 16.0, 16.0)


def _inject_tails(values: torch.Tensor, profile: Profile, generator: torch.Generator) -> None:
    if profile.tail_probability > 0:
        mask = torch.rand(values.shape, generator=generator) < profile.tail_probability
        values[mask] *= profile.tail_multiplier
    if profile.sparsity > 0:
        mask = torch.rand(values.shape, generator=generator) < profile.sparsity
        values[mask] = 0.0


def _raw_weight(
    out_features: int,
    in_features: int,
    profile: Profile,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    column_scale = _channel_scales(in_features, profile.channel_spread, generator)
    weight = torch.randn(out_features, in_features, generator=generator) / math.sqrt(in_features)
    if profile.weight_rank:
        rank = min(profile.weight_rank, out_features, in_features)
        left = torch.randn(out_features, rank, generator=generator)
        right = torch.randn(rank, in_features, generator=generator)
        correlated = left @ right / math.sqrt(rank * in_features)
        weight.add_(correlated, alpha=profile.correlated_strength)
    weight *= column_scale
    _inject_tails(weight, profile, generator)
    if profile.block_spike:
        block_count = in_features // 64
        chosen = torch.randperm(block_count, generator=generator)[:max(1, block_count // 8)]
        weight[:, chosen.repeat_interleave(64) * 64 + torch.arange(64).repeat(len(chosen))] *= profile.block_spike
    return weight, column_scale


def _raw_activation(
    rows: int,
    channels: int,
    column_scale: torch.Tensor,
    profile: Profile,
    generator: torch.Generator,
    test: bool,
    shared_basis: torch.Tensor,
    shared_test_drift: torch.Tensor,
) -> torch.Tensor:
    values = torch.randn(rows, channels, generator=generator)
    # Anti-correlated A/W ranges make SmoothQuant and output-aware searches useful.
    scales = column_scale.rsqrt()
    if test and profile.test_drift:
        scales = scales * shared_test_drift
    values *= scales
    if shared_basis.numel():
        # The five calibration and five test tensors belonging to one Weight
        # are different samples from the same activation distribution.  Keep
        # the latent coefficients private to each tensor, but share its basis.
        # This preserves the stable cross-64-channel covariance revealed by
        # the online feedback-step ablation without leaking any activation.
        latent_rank = int(shared_basis.shape[0])
        latent = torch.randn(rows, latent_rank, generator=generator)
        values.add_(
            latent @ shared_basis / math.sqrt(latent_rank),
            alpha=profile.activation_strength,
        )
    _inject_tails(values, profile, generator)
    return values


def build_linear_group(
    profile: Profile,
    seed: int,
    base: int,
    rows: tuple[int, ...],
) -> dict:
    generator = torch.Generator().manual_seed(seed)
    out_features, in_features = _shape(profile, base)
    weight, column_scale = _raw_weight(out_features, in_features, profile, generator)
    activation_rank = min(profile.activation_rank, in_features)
    if activation_rank:
        # One basis spans every 64-channel HiF4 block.  Row normalization keeps
        # activation_strength comparable across correlation/full presets.
        shared_basis = torch.randn(
            activation_rank, in_features, generator=generator
        )
        shared_basis /= shared_basis.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
    else:
        shared_basis = torch.empty(0, in_features)
    shared_test_drift = torch.exp(
        torch.randn(in_features, generator=generator) * profile.test_drift
    ).clamp(0.25, 4.0)
    group = {
        "key": f"linear/{profile.name}/{seed}",
        "profile": profile.name,
        "logical_strides": list(STRIDES),
        "weight": quantize_nvfp4(weight),
        "calib_activation_list": [],
        "test_activation_list": [],
    }
    del weight
    for count in rows:
        group["calib_activation_list"].append(quantize_nvfp4(
            _raw_activation(
                count, in_features, column_scale, profile, generator, False,
                shared_basis, shared_test_drift,
            )
        ))
    for count in rows:
        group["test_activation_list"].append(quantize_nvfp4(
            _raw_activation(
                count, in_features, column_scale, profile, generator, True,
                shared_basis, shared_test_drift,
            )
        ))
    return group


def _attention_sample(
    rows: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    generator: torch.Generator,
    drift: float,
    fixed_channel: torch.Tensor | None = None,
) -> dict[str, list[torch.Tensor]]:
    replicas = q_heads // kv_heads
    channel = fixed_channel
    if channel is None:
        channel = torch.exp(
            torch.randn(kv_heads, head_dim, generator=generator) * 0.8
        ).clamp(0.125, 8.0)
    if drift and fixed_channel is None:
        channel *= torch.exp(torch.randn(channel.shape, generator=generator) * drift).clamp(0.5, 2.0)
    latent = torch.randn(rows, kv_heads, head_dim, generator=generator)
    k = latent * channel
    q = latent.repeat_interleave(replicas, dim=1) / channel.repeat_interleave(replicas, dim=0)
    q_noise = 0.05 if fixed_channel is not None else 0.45
    q += q_noise * torch.randn(q.shape, generator=generator)
    v = 0.55 * latent + torch.randn(latent.shape, generator=generator)
    for tensor in (q, k, v):
        mask = torch.rand(tensor.shape, generator=generator) < 0.003
        tensor[mask] *= 10.0
    return {
        "q": quantize_nvfp4(q.reshape(rows, -1)),
        "k": quantize_nvfp4(k.reshape(rows, -1)),
        "v": quantize_nvfp4(v.reshape(rows, -1)),
    }


def build_attention_group(
    seed: int,
    rows: tuple[int, ...],
    full: bool,
    same_model: bool = False,
) -> dict:
    generator = torch.Generator().manual_seed(seed)
    # Keep head_dim=256 even in the correlation preset so v8's multi-block
    # zigzag candidate and the online RRS ablations are actually exercised.
    q_heads, kv_heads, head_dim = ((16, 4, 256) if full else (8, 2, 256))
    fixed_channel = None
    if same_model:
        fixed_channel = torch.exp(
            torch.randn(kv_heads, head_dim, generator=generator) * 0.8
        ).clamp(0.125, 8.0)
    return {
        "key": f"attention/synthetic/{seed}",
        "q_num_heads": q_heads,
        "kv_num_heads": kv_heads,
        "head_dim": head_dim,
        "attn_type": "gqa",
        "logical_strides": list(STRIDES),
        "calib": [
            _attention_sample(
                n, q_heads, kv_heads, head_dim, generator, 0.0,
                fixed_channel,
            )
            for n in rows
        ],
        "test": [
            _attention_sample(
                n, q_heads, kv_heads, head_dim, generator, 0.20,
                fixed_channel,
            )
            for n in rows
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("correlation", "full"), default="correlation")
    parser.add_argument("--output", type=Path, default=Path("local_fitted"))
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--activation-strength", type=float,
        help="override the shared cross-channel latent strength for every profile",
    )
    parser.add_argument(
        "--test-drift-scale", type=float, default=0.5,
        help="multiply every profile's calibration/test diagonal drift",
    )
    parser.add_argument(
        "--linear-only", action="store_true",
        help="build only Linear data (useful for the multi-GB full preset)",
    )
    parser.add_argument(
        "--full-linear-rows", action="store_true",
        help="use 1/10/128/512/1024 rows at correlation width as online-like calibration",
    )
    parser.add_argument(
        "--linear-shape-mix", choices=("mixed", "rectangular"), default="mixed",
        help="select the historical mixed bank or the post-v278 rectangular hypothesis",
    )
    parser.add_argument(
        "--attention-profile", choices=("legacy", "same_model"),
        default="legacy",
        help="keep historical per-sample channels or share one fixed Q/K projection per group",
    )
    args = parser.parse_args()

    full = args.preset == "full"
    base = 4096 if full else 512
    rows = (
        STRIDES
        if (full or args.full_linear_rows)
        else (1, 10, 32, 64, 128)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    source_profiles = (
        RECTANGULAR_PROFILES
        if args.linear_shape_mix == "rectangular"
        else PROFILES
    )
    profiles = tuple(
        replace(
            profile,
            activation_strength=(
                args.activation_strength
                if args.activation_strength is not None
                else profile.activation_strength
            ),
            test_drift=profile.test_drift * args.test_drift_scale,
        )
        for profile in source_profiles
    )

    # One instance of each profile gives ten weights x five test activations.
    linear = [
        build_linear_group(profile, args.seed + index, base, rows)
        for index, profile in enumerate(profiles)
    ]
    # Fifty groups x five tests matches the post-2026-08-31 250 Attention tests.
    attention = [] if args.linear_only else [
        build_attention_group(
            args.seed + 10000 + repeat * 101 + offset,
            rows,
            full,
            same_model=(args.attention_profile == "same_model"),
        )
        for repeat in range(5)
        for offset in ATTENTION_SEED_OFFSETS
    ]
    torch.save(linear, args.output / "linear.pt")
    if not args.linear_only:
        torch.save(attention, args.output / "attn.pt")
    manifest = {
        "preset": args.preset,
        "seed": args.seed,
        "linear_weight_groups": len(linear),
        "linear_calibration_activations": sum(len(g["calib_activation_list"]) for g in linear),
        "linear_test_activations": sum(len(g["test_activation_list"]) for g in linear),
        "attention_groups": len(attention),
        "attention_test_datasets": sum(len(g["test"]) for g in attention),
        "attention_seed_offsets": list(ATTENTION_SEED_OFFSETS),
        "logical_strides": list(STRIDES),
        "stored_rows": list(rows),
        "profiles": [asdict(profile) for profile in profiles],
        "activation_strength_override": args.activation_strength,
        "test_drift_scale": args.test_drift_scale,
        "linear_only": args.linear_only,
        "full_linear_rows": args.full_linear_rows,
        "linear_shape_mix": args.linear_shape_mix,
        "attention_profile": args.attention_profile,
        "distribution_source": "online ablation constraints; mini_sample schema only",
        "online_feedback_revision": 3,
        "activation_covariance": (
            "per-weight shared latent basis; independent sample coefficients; "
            "shared calibration/test covariance and test-domain diagonal drift"
        ),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
