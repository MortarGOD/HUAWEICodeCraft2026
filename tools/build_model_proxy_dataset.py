#!/usr/bin/env python3
"""Build a HiF4 proxy dataset shaped like Qwen3.6 and DeepSeek-V4.

Checkpoint mode contains real Weight/QKV projection tensors, but the hidden
states are modeled, not captured full-model activations. It pins public model
revisions and reproduces their projection dimensions, GQA
layout, head dimensions, RMS-normalized activation statistics, correlated
Q/K channels, partial RoPE, trained-weight-like channel imbalance, and the
contest's current 50 Linear / 250 Attention case mix.

The default ``covariance`` preset preserves a group-level semantic covariance
and actual row layouts. ``correlation`` retains the historical counterexample. The
``online-shape`` preset uses model/TP-local dimensions and the exact logical
row counts 1/10/128/512/1024; it is much larger.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from urllib.request import Request, urlopen

import torch

from build_fitted_dataset import STRIDES, quantize_nvfp4


@dataclass(frozen=True)
class LinearLayer:
    name: str
    out_features: int
    in_features: int
    channel_spread: float
    low_rank_strength: float
    tensor_name: str
    shard: str
    scale_name: str | None = None


@dataclass(frozen=True)
class ModelProfile:
    name: str
    repo_id: str
    revision: str
    hidden_size: int
    initializer_range: float
    q_heads: int
    kv_heads: int
    head_dim: int
    tp_size: int
    rotary_dim: int
    attention_noise: float
    linear_layers: tuple[LinearLayer, ...]


QWEN36 = ModelProfile(
    name="qwen3.6-35b-a3b",
    repo_id="Qwen/Qwen3.6-35B-A3B",
    revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
    hidden_size=2048,
    initializer_range=0.02,
    q_heads=16,
    kv_heads=2,
    head_dim=256,
    tp_size=1,
    rotary_dim=64,
    attention_noise=0.11,
    linear_layers=(
        # The checkpoint packs Q and the attention output gate together.  The
        # [8192, 2048] tensor exactly matches the public mini_sample Weight;
        # its runtime Q is split back to 16 * 256 channels.
        LinearLayer("full_attn.q_proj_gated", 8192, 2048, 0.52, 0.18,
                    "model.language_model.layers.3.self_attn.q_proj.weight", "model-00003-of-00026.safetensors"),
        LinearLayer("full_attn.o_proj", 2048, 4096, 0.58, 0.20,
                    "model.language_model.layers.3.self_attn.o_proj.weight", "model-00003-of-00026.safetensors"),
        LinearLayer("shared_expert.gate_proj", 512, 2048, 0.70, 0.26,
                    "model.language_model.layers.3.mlp.shared_expert.gate_proj.weight", "model-00003-of-00026.safetensors"),
        LinearLayer("shared_expert.up_proj", 512, 2048, 0.65, 0.24,
                    "model.language_model.layers.3.mlp.shared_expert.up_proj.weight", "model-00003-of-00026.safetensors"),
        LinearLayer("shared_expert.down_proj", 2048, 512, 0.78, 0.30,
                    "model.language_model.layers.3.mlp.shared_expert.down_proj.weight", "model-00003-of-00026.safetensors"),
    ),
)


DEEPSEEK_V4_FLASH = ModelProfile(
    name="deepseek-v4-flash-0731",
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    revision="7872f01b1d1fe23eabc4c98b48bffcef5a386062",
    hidden_size=4096,
    initializer_range=0.02,
    q_heads=64,
    kv_heads=1,
    head_dim=512,
    # Online inference is expected to expose a TP-local Q projection.  At TP=8
    # this gives 8 Q heads and one replicated KV head.
    tp_size=8,
    rotary_dim=64,
    attention_noise=0.08,
    linear_layers=(
        LinearLayer("attn.wq_a", 1024, 4096, 0.62, 0.22,
                    "layers.0.attn.wq_a.weight", "model-00002-of-00048.safetensors", "layers.0.attn.wq_a.scale"),
        LinearLayer("attn.wq_b.tp8", 4096, 1024, 0.72, 0.28,
                    "layers.0.attn.wq_b.weight", "model-00002-of-00048.safetensors", "layers.0.attn.wq_b.scale"),
        LinearLayer("attn.wo_b.tp8", 4096, 1024, 0.66, 0.24,
                    "layers.0.attn.wo_b.weight", "model-00002-of-00048.safetensors", "layers.0.attn.wo_b.scale"),
        LinearLayer("shared_expert.w1", 2048, 4096, 0.82, 0.34,
                    "layers.0.ffn.shared_experts.w1.weight", "model-00002-of-00048.safetensors", "layers.0.ffn.shared_experts.w1.scale"),
        LinearLayer("shared_expert.w2", 4096, 2048, 0.88, 0.38,
                    "layers.0.ffn.shared_experts.w2.weight", "model-00002-of-00048.safetensors", "layers.0.ffn.shared_experts.w2.scale"),
    ),
)


MODELS = (QWEN36, DEEPSEEK_V4_FLASH)
_ATTENTION_PROJECTION_CACHE: dict[tuple[str, str], dict[str, torch.Tensor]] = {}


def _round64(value: float, minimum: int = 64) -> int:
    return max(minimum, int(round(value / 64.0)) * 64)


def _linear_shape(layer: LinearLayer, preset: str) -> tuple[int, int]:
    if preset == "online-shape":
        return layer.out_features, layer.in_features
    # Cap only the largest dimension and apply one common factor to both axes.
    # This preserves the exact up/down aspect ratio: using independent minima
    # previously collapsed Qwen's 512x2048 experts into misleading squares.
    factor = min(1.0, 2048.0 / max(layer.out_features, layer.in_features))
    return _round64(layer.out_features * factor), _round64(layer.in_features * factor)


def _rms_normalize(values: torch.Tensor) -> torch.Tensor:
    return values / values.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)


def _channel_scale(width: int, spread: float, generator: torch.Generator) -> torch.Tensor:
    scale = torch.exp(torch.randn(width, generator=generator) * spread)
    return (scale / scale.log().mean().exp()).clamp(0.125, 8.0)


def _trained_like_weight(
    rows: int,
    cols: int,
    layer: LinearLayer,
    model: ModelProfile,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    column = _channel_scale(cols, layer.channel_spread, generator)
    weight = torch.randn(rows, cols, generator=generator) * model.initializer_range
    rank = min(16, rows, cols)
    left = torch.randn(rows, rank, generator=generator)
    right = torch.randn(rank, cols, generator=generator)
    weight.add_(
        left @ right,
        alpha=model.initializer_range * layer.low_rank_strength / math.sqrt(rank),
    )
    weight *= column
    # Sparse trained-weight-like outliers distinguish clipping from rotation.
    outliers = torch.rand(rows, cols, generator=generator) < 0.0015
    weight[outliers] *= 7.0
    return weight, column


def _read_range(url: str, start: int, end: int) -> bytes:
    error: Exception | None = None
    for attempt in range(8):
        try:
            request = Request(url, headers={
                "Range": f"bytes={start}-{end}",
                "User-Agent": "hif4-model-proxy/1.0",
                "Connection": "close",
            })
            with urlopen(request, timeout=90) as response:
                payload = response.read()
            expected = end - start + 1
            if len(payload) != expected:
                raise RuntimeError(f"range returned {len(payload)} bytes, expected {expected}")
            return payload
        except Exception as exc:  # network retries are deliberately bounded
            error = exc
            time.sleep(min(5.0, 1.0 + attempt))
    raise RuntimeError(f"failed to fetch {url} bytes {start}:{end}: {error}")


@lru_cache(maxsize=None)
def _safetensor_header(model: ModelProfile, shard: str) -> tuple[dict, str, int]:
    url = (
        f"https://huggingface.co/{model.repo_id}/resolve/"
        f"{model.revision}/{shard}"
    )
    header_size = struct.unpack("<Q", _read_range(url, 0, 7))[0]
    header = json.loads(_read_range(url, 8, 7 + header_size))
    return header, url, 8 + header_size


def _decode_tensor(raw: bytes, metadata: dict) -> torch.Tensor:
    dtype_name = metadata["dtype"]
    dtype_map = {
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
        "F8_E4M3": torch.float8_e4m3fn,
        "I8": torch.int8,
        "U8": torch.uint8,
    }
    if dtype_name == "F8_E8M0":
        # Unsigned exponent-only scale: 2 ** (stored_exponent - 127).
        encoded = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
        return torch.pow(2.0, encoded.float() - 127.0).reshape(metadata["shape"])
    if dtype_name not in dtype_map:
        raise ValueError(f"unsupported safetensors dtype {dtype_name}")
    return torch.frombuffer(bytearray(raw), dtype=dtype_map[dtype_name]).clone().reshape(
        metadata["shape"]
    )


def _fetch_checkpoint_tensor(
    model: ModelProfile,
    layer: LinearLayer,
    tensor_name: str,
) -> torch.Tensor:
    header, url, data_start = _safetensor_header(model, layer.shard)
    metadata = header[tensor_name]
    start, end = metadata["data_offsets"]
    raw = _read_range(url, data_start + start, data_start + end - 1)
    return _decode_tensor(raw, metadata)


def _checkpoint_weight(
    model: ModelProfile,
    layer: LinearLayer,
    rows: int,
    cols: int,
    cache_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = f"{model.name}--{layer.name.replace('.', '_')}.pt"
    cache_path = cache_dir / cache_name
    if cache_path.exists():
        source = torch.load(cache_path, map_location="cpu", weights_only=True)
    else:
        source = _fetch_checkpoint_tensor(model, layer, layer.tensor_name)
        if layer.scale_name is not None:
            scale = _fetch_checkpoint_tensor(model, layer, layer.scale_name).float()
            source = source.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
        else:
            source = source.float()
        torch.save(source.to(torch.bfloat16), cache_path)
    source = source.float()
    row_indices = torch.linspace(0, source.shape[0] - 1, rows).round().long()
    col_indices = torch.linspace(0, source.shape[1] - 1, cols).round().long()
    weight = source.index_select(0, row_indices).index_select(1, col_indices).contiguous()
    column = weight.square().mean(dim=0).sqrt().clamp_min(1e-8)
    column = (column / column.log().mean().exp()).clamp(0.125, 8.0)
    return weight, column


def _checkpoint_matrix(
    model: ModelProfile,
    name: str,
    shard: str,
    scale_name: str | None,
    cache_dir: Path,
    cache_tag: str,
) -> torch.Tensor:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{model.name}--{cache_tag}.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=True).float()
    descriptor = LinearLayer(cache_tag, 64, 64, 0.0, 0.0, name, shard, scale_name)
    matrix = _fetch_checkpoint_tensor(model, descriptor, name)
    if scale_name is not None:
        scale = _fetch_checkpoint_tensor(model, descriptor, scale_name).float()
        matrix = matrix.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    else:
        matrix = matrix.float()
    torch.save(matrix.to(torch.bfloat16), cache_path)
    return matrix


def _attention_projections(
    model: ModelProfile, cache_dir: Path
) -> dict[str, torch.Tensor]:
    key = (model.name, str(cache_dir.resolve()))
    if key in _ATTENTION_PROJECTION_CACHE:
        return _ATTENTION_PROJECTION_CACHE[key]
    if model.name.startswith("qwen"):
        shard = "model-00003-of-00026.safetensors"
        bundle = {
            "q": _checkpoint_matrix(
                model, "model.language_model.layers.3.self_attn.q_proj.weight",
                shard, None, cache_dir, "full_attn_q_proj_gated",
            ),
            "k": _checkpoint_matrix(
                model, "model.language_model.layers.3.self_attn.k_proj.weight",
                shard, None, cache_dir, "full_attn_k_proj",
            ),
            "v": _checkpoint_matrix(
                model, "model.language_model.layers.3.self_attn.v_proj.weight",
                shard, None, cache_dir, "full_attn_v_proj",
            ),
            "q_norm": _checkpoint_matrix(
                model, "model.language_model.layers.3.self_attn.q_norm.weight",
                shard, None, cache_dir, "full_attn_q_norm",
            ),
            "k_norm": _checkpoint_matrix(
                model, "model.language_model.layers.3.self_attn.k_norm.weight",
                shard, None, cache_dir, "full_attn_k_norm",
            ),
        }
    else:
        shard = "model-00002-of-00048.safetensors"
        bundle = {
            "q_a": _checkpoint_matrix(
                model, "layers.0.attn.wq_a.weight", shard,
                "layers.0.attn.wq_a.scale", cache_dir, "attn_wq_a",
            ),
            "q_b": _checkpoint_matrix(
                model, "layers.0.attn.wq_b.weight", shard,
                "layers.0.attn.wq_b.scale", cache_dir, "attn_wq_b_tp8",
            ),
            "kv": _checkpoint_matrix(
                model, "layers.0.attn.wkv.weight", shard,
                "layers.0.attn.wkv.scale", cache_dir, "attn_wkv",
            ),
            "q_norm": _checkpoint_matrix(
                model, "layers.0.attn.q_norm.weight", shard,
                None, cache_dir, "attn_q_norm",
            ),
            "kv_norm": _checkpoint_matrix(
                model, "layers.0.attn.kv_norm.weight", shard,
                None, cache_dir, "attn_kv_norm",
            ),
        }
    _ATTENTION_PROJECTION_CACHE[key] = bundle
    return bundle


def _linear_activation(
    rows: int,
    cols: int,
    column: torch.Tensor,
    shared_basis: torch.Tensor,
    drift: torch.Tensor,
    generator: torch.Generator,
    test: bool,
) -> torch.Tensor:
    values = _rms_normalize(torch.randn(rows, cols, generator=generator))
    values *= column.rsqrt()
    latent = torch.randn(rows, shared_basis.shape[0], generator=generator)
    values.add_(latent @ shared_basis / math.sqrt(shared_basis.shape[0]), alpha=0.32)
    if test:
        values *= drift
    mask = torch.rand(rows, cols, generator=generator) < 0.001
    values[mask] *= 8.0
    return values


def build_linear_group(
    model: ModelProfile,
    layer: LinearLayer,
    seed: int,
    preset: str,
    rows: tuple[int, ...],
    weight_source: str,
    tensor_cache: Path,
) -> dict:
    generator = torch.Generator().manual_seed(seed)
    out_features, in_features = _linear_shape(layer, preset)
    if weight_source == "checkpoint":
        weight, column = _checkpoint_weight(
            model, layer, out_features, in_features, tensor_cache
        )
    else:
        weight, column = _trained_like_weight(
            out_features, in_features, layer, model, generator
        )
    basis = torch.randn(16, in_features, generator=generator)
    basis = _rms_normalize(basis)
    drift = torch.exp(torch.randn(in_features, generator=generator) * 0.07).clamp(0.7, 1.4)
    group = {
        "key": f"linear/{model.name}/{layer.name}/{seed}",
        "profile": f"{model.name}:{layer.name}",
        "source_model": model.repo_id,
        "source_revision": model.revision,
        "weight_source": weight_source,
        "source_tensor": layer.tensor_name,
        "logical_strides": list(STRIDES),
        "weight": quantize_nvfp4(weight),
        "calib_activation_list": [],
        "test_activation_list": [],
    }
    for count in rows:
        group["calib_activation_list"].append(
            quantize_nvfp4(
                _linear_activation(
                    count, in_features, column, basis, drift, generator, False
                )
            )
        )
    for count in rows:
        group["test_activation_list"].append(
            quantize_nvfp4(
                _linear_activation(
                    count, in_features, column, basis, drift, generator, True
                )
            )
        )
    return group


def _apply_partial_rope(values: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    rotary_dim = min(rotary_dim, values.shape[-1])
    rotary_dim -= rotary_dim % 2
    if rotary_dim == 0:
        return values
    rows = values.shape[0]
    angles = torch.arange(rows, dtype=torch.float32).view(rows, 1, 1)
    inv = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, rotary_dim, 2, dtype=torch.float32)
        / rotary_dim
    ).view(1, 1, -1)
    cosine, sine = torch.cos(angles * inv), torch.sin(angles * inv)
    rotated = values.clone()
    even, odd = values[..., :rotary_dim:2], values[..., 1:rotary_dim:2]
    rotated[..., :rotary_dim:2] = even * cosine - odd * sine
    rotated[..., 1:rotary_dim:2] = even * sine + odd * cosine
    return rotated


def _attention_layout(model: ModelProfile, preset: str) -> tuple[int, int, int]:
    if preset == "online-shape":
        return max(1, model.q_heads // model.tp_size), model.kv_heads, model.head_dim
    q_heads = 8
    kv_heads = min(model.kv_heads, q_heads)
    while q_heads % kv_heads:
        q_heads -= 1
    return q_heads, kv_heads, model.head_dim


def _attention_sample(
    rows: int,
    model: ModelProfile,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    channel: torch.Tensor,
    generator: torch.Generator,
    test: bool,
) -> dict[str, list[torch.Tensor]]:
    repeats = q_heads // kv_heads
    latent = _rms_normalize(
        torch.randn(rows, kv_heads, head_dim, generator=generator)
    )
    current_channel = channel
    if test:
        current_channel = channel * torch.exp(
            torch.randn(channel.shape, generator=generator) * 0.035
        ).clamp(0.8, 1.25)
    k = latent * current_channel
    q = latent.repeat_interleave(repeats, dim=1)
    q /= current_channel.repeat_interleave(repeats, dim=0)
    q += model.attention_noise * torch.randn(q.shape, generator=generator)
    k += model.attention_noise * torch.randn(k.shape, generator=generator)
    q = _apply_partial_rope(q, model.rotary_dim)
    k = _apply_partial_rope(k, model.rotary_dim)
    v = 0.65 * latent + 0.75 * torch.randn(latent.shape, generator=generator)
    for tensor in (q, k, v):
        mask = torch.rand(tensor.shape, generator=generator) < 0.0015
        tensor[mask] *= 8.0
    return {
        "q": quantize_nvfp4(q.reshape(rows, -1)),
        "k": quantize_nvfp4(k.reshape(rows, -1)),
        "v": quantize_nvfp4(v.reshape(rows, -1)),
    }


def _proxy_hidden(
    rows: int, width: int, generator: torch.Generator,
    persistent_basis: torch.Tensor | None = None,
    structured_strength: float = 0.0,
) -> torch.Tensor:
    hidden = torch.randn(rows, width, generator=generator)
    # Transformer residual streams have persistent token and low-rank semantic
    # structure.  Keep it moderate so different samples remain independent.
    if rows > 1:
        for index in range(1, rows):
            hidden[index].add_(hidden[index - 1], alpha=0.35)
    latent = torch.randn(rows, 8, generator=generator)
    if persistent_basis is None:
        basis = torch.randn(8, width, generator=generator)
        hidden.add_(latent @ basis / math.sqrt(8.0), alpha=0.18)
    else:
        # Independent tokens/prompts share a *population* covariance. Basis
        # directions are sampled real checkpoint rows, not test activations.
        rank = persistent_basis.shape[0]
        latent = torch.randn(rows, rank, generator=generator)
        hidden.add_(latent @ persistent_basis / math.sqrt(rank),
                    alpha=structured_strength)
    return _rms_normalize(hidden)


def _attention_sample_projected(
    rows: int,
    model: ModelProfile,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    projections: dict[str, torch.Tensor],
    generator: torch.Generator,
    context_drift: float,
    persistent_basis: torch.Tensor | None = None,
    structured_strength: float = 0.0,
) -> dict[str, list[torch.Tensor]]:
    hidden = _proxy_hidden(rows, model.hidden_size, generator,
                           persistent_basis, structured_strength)
    if model.name.startswith("qwen"):
        fused = hidden @ projections["q"].T
        fused = fused.view(rows, model.q_heads, 2 * head_dim)
        q = fused[..., :head_dim][:, :q_heads]
        k = (hidden @ projections["k"].T).view(rows, model.kv_heads, head_dim)
        v = (hidden @ projections["v"].T).view(rows, model.kv_heads, head_dim)
        q = _rms_normalize(q) * projections["q_norm"].view(1, 1, -1)
        k = _rms_normalize(k) * projections["k_norm"].view(1, 1, -1)
    else:
        q_latent = _rms_normalize(hidden @ projections["q_a"].T)
        q_latent *= projections["q_norm"].view(1, -1)
        q = (q_latent @ projections["q_b"].T).view(
            rows, model.q_heads, head_dim
        )[:, :q_heads]
        q = _rms_normalize(q)
        kv = _rms_normalize(hidden @ projections["kv"].T)
        kv *= projections["kv_norm"].view(1, -1)
        kv = kv.view(
            rows, kv_heads, head_dim
        )
        k = kv
        v = kv
    q = _apply_partial_rope(q, model.rotary_dim)
    k = _apply_partial_rope(k, model.rotary_dim)
    if context_drift > 0:
        # Real prompts activate different persistent outlier channels even for
        # one fixed projection.  This context-specific drift is the smallest
        # fitted axis that reproduces v41 rotation/GPTQ's large online gain.
        q_scale = torch.exp(
            torch.randn(q_heads * head_dim, generator=generator) * context_drift
        ).clamp(0.05, 20.0)
        k_scale = torch.exp(
            torch.randn(kv_heads * head_dim, generator=generator) * context_drift
        ).clamp(0.05, 20.0)
        q = q * q_scale.view(1, q_heads, head_dim)
        k = k * k_scale.view(1, kv_heads, head_dim)
    if not model.name.startswith("qwen"):
        # DeepSeek-V4 explicitly uses the same RoPE-transformed tensor as K/V.
        v = k
    return {
        "q": quantize_nvfp4(q.reshape(rows, -1)),
        "k": quantize_nvfp4(k.reshape(rows, -1)),
        "v": quantize_nvfp4(v.reshape(rows, -1)),
    }


def build_attention_group(
    model: ModelProfile,
    seed: int,
    preset: str,
    rows: tuple[int, ...],
    attention_source: str,
    tensor_cache: Path,
    context_drift: float,
    covariance_rank: int = 0,
    structured_strength: float = 0.0,
) -> dict:
    generator = torch.Generator().manual_seed(seed)
    q_heads, kv_heads, head_dim = _attention_layout(model, preset)
    projections = (
        _attention_projections(model, tensor_cache)
        if attention_source == "checkpoint-projection" else None
    )
    channel = torch.exp(
        torch.randn(kv_heads, head_dim, generator=generator) * 0.62
    ).clamp(0.125, 8.0)
    persistent_basis = None
    if covariance_rank:
        if projections is None:
            raise ValueError("persistent covariance requires checkpoint projections")
        source = projections["q"] if model.name.startswith("qwen") else projections["q_a"]
        count = min(covariance_rank, source.shape[0])
        indices = torch.randperm(source.shape[0], generator=generator)[:count]
        persistent_basis = _rms_normalize(source[indices].float()).contiguous()

    def sample(count: int, test: bool) -> dict[str, list[torch.Tensor]]:
        if projections is not None:
            return _attention_sample_projected(
                count, model, q_heads, kv_heads, head_dim, projections, generator,
                context_drift, persistent_basis, structured_strength,
            )
        return _attention_sample(
            count, model, q_heads, kv_heads, head_dim, channel, generator, test
        )
    return {
        "key": f"attention/{model.name}/{seed}",
        "source_model": model.repo_id,
        "source_revision": model.revision,
        "attention_source": attention_source,
        "covariance_rank": covariance_rank,
        "structured_strength": structured_strength,
        "q_num_heads": q_heads,
        "kv_num_heads": kv_heads,
        "head_dim": head_dim,
        "attn_type": "gqa",
        "logical_strides": list(STRIDES),
        "calib": [sample(count, False) for count in rows],
        "test": [sample(count, True) for count in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset", choices=("covariance", "correlation", "online-shape"), default="covariance"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("local_fitted_models_covariance_v1")
    )
    parser.add_argument("--seed", type=int, default=20262903)
    parser.add_argument(
        "--weight-source", choices=("synthetic", "checkpoint"), default="checkpoint",
        help="range-fetch pinned checkpoint tensors for real weights",
    )
    parser.add_argument(
        "--tensor-cache", type=Path, default=Path("local_model_tensor_cache"),
    )
    parser.add_argument(
        "--attention-source",
        choices=("synthetic", "checkpoint-projection"),
        default="checkpoint-projection",
        help="derive Q/K/V from real checkpoint projections and proxy hidden states",
    )
    parser.add_argument(
        "--attention-context-drift", type=float, default=None,
        help="per-prompt log-channel drift fitted to current online Attention ablations",
    )
    parser.add_argument(
        "--groups-per-model", type=int, default=25,
        help="25 gives 50 Attention groups and 250 tests",
    )
    parser.add_argument("--linear-only", action="store_true")
    parser.add_argument("--attention-covariance-rank", type=int, default=None)
    parser.add_argument("--attention-structured-strength", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="explicitly replace generated files in the selected output directory")
    parser.add_argument("--rows", type=str, default=None,
                        help="five stored row counts; unset keeps preset defaults")
    parser.add_argument("--attention-only", action="store_true")
    args = parser.parse_args()
    if args.attention_covariance_rank is None:
        args.attention_covariance_rank = 128 if args.preset == "covariance" else 0
    if args.attention_structured_strength is None:
        args.attention_structured_strength = 4.0 if args.preset == "covariance" else 0.0
    if args.attention_context_drift is None:
        args.attention_context_drift = 0.05 if args.preset == "covariance" else 0.8
    if not args.overwrite and any((args.output / f).exists() for f in ('linear.pt','attn.pt')):
        parser.error("output already contains data; choose a new directory or explicitly --overwrite")
    if args.linear_only and args.attention_only:
        parser.error("--linear-only and --attention-only are mutually exclusive")
    if args.groups_per_model < 1:
        parser.error("--groups-per-model must be positive")

    rows = (tuple(int(x) for x in args.rows.split(",")) if args.rows else
            (STRIDES if args.preset in ("online-shape", "covariance") else (1, 10, 32, 64, 128)))
    if len(rows) != 5 or any(x < 1 for x in rows):
        parser.error("--rows must contain five positive integers")
    if args.attention_covariance_rank < 0 or args.attention_structured_strength < 0:
        parser.error("covariance rank and strength must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    linear = [] if args.attention_only else [
        build_linear_group(
            model, layer, args.seed + model_index * 100 + layer_index,
            args.preset, rows, args.weight_source, args.tensor_cache,
        )
        for model_index, model in enumerate(MODELS)
        for layer_index, layer in enumerate(model.linear_layers)
    ]
    attention = [] if args.linear_only else [
        build_attention_group(
            model,
            args.seed + 10000 + model_index * 1000 + group_index * 37,
            args.preset,
            rows,
            args.attention_source,
            args.tensor_cache,
            args.attention_context_drift,
            args.attention_covariance_rank,
            args.attention_structured_strength,
        )
        for model_index, model in enumerate(MODELS)
        for group_index in range(args.groups_per_model)
    ]

    if linear:
        torch.save(linear, args.output / "linear.pt")
    if attention:
        torch.save(attention, args.output / "attn.pt")
    manifest = {
        "kind": (
            "checkpoint weights + architecture/statistics activation proxy"
            if args.weight_source == "checkpoint"
            else "architecture-and-statistics proxy; no extracted model tensors"
        ),
        "preset": args.preset,
        "seed": args.seed,
        "weight_source": args.weight_source,
        "attention_source": args.attention_source,
        "attention_context_drift": args.attention_context_drift,
        "attention_covariance_rank": args.attention_covariance_rank,
        "attention_structured_strength": args.attention_structured_strength,
        "stored_rows": list(rows),
        "logical_strides": list(STRIDES),
        "linear_weight_groups": len(linear),
        "linear_test_activations": sum(
            len(group["test_activation_list"]) for group in linear
        ),
        "attention_groups": len(attention),
        "attention_test_datasets": sum(len(group["test"]) for group in attention),
        "models": [asdict(model) for model in MODELS],
        "deepseek_attention_assumption": (
            "8-way tensor-parallel local Q heads; KV head replicated"
        ),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
