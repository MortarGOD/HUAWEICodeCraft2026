"""Raw quality/timing diagnostics; local scores are not online predictions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch

from hif4_quant import dequantize_hif4, dequantize_nvfp4, quantize_to_hif4


def load_solution(path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("candidate_solution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mse(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.mean((left.float() - right.float()).square()).item())


def score(baseline_mse: float, player_mse: float) -> float:
    return (baseline_mse - player_mse) / baseline_mse


def gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    seq = int(q.shape[0])
    qh = q.float().reshape(seq, q_heads, head_dim).transpose(0, 1)
    kh = k.float().reshape(seq, kv_heads, head_dim).transpose(0, 1)
    vh = v.float().reshape(seq, kv_heads, head_dim).transpose(0, 1)
    repeats = q_heads // kv_heads
    kh = kh.repeat_interleave(repeats, dim=0)
    vh = vh.repeat_interleave(repeats, dim=0)
    probs = torch.softmax(torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(head_dim), dim=-1)
    return torch.matmul(probs, vh).transpose(0, 1).reshape(seq, q_heads * head_dim)


def evaluate_linear(solution, groups: list) -> tuple[list[float], float]:
    all_scores: list[float] = []
    started = time.perf_counter()
    for group_idx, group in enumerate(groups):
        wq, ws = group["weight"]
        weight = dequantize_nvfp4(wq, ws).float()
        baseline_wp = quantize_to_hif4(weight)
        baseline_w = dequantize_hif4(baseline_wp, tuple(weight.shape))

        result = solution.hif4_calibration_and_quantize_weight(
            wq, ws, group["calib_activation_list"]
        )
        player_w = dequantize_hif4(result["weight_params"], tuple(weight.shape))
        for test_idx, (aq, ass) in enumerate(group["test_activation_list"]):
            activation = dequantize_nvfp4(aq, ass).float()
            reference = torch.matmul(activation, weight.transpose(-1, -2))
            baseline_a = dequantize_hif4(quantize_to_hif4(activation), tuple(activation.shape))
            baseline_out = torch.matmul(baseline_a, baseline_w.transpose(-1, -2))
            player_ap = solution.hif4_dynamic_quantize_activation(
                aq, ass, result["activation_state"]
            )
            player_a = dequantize_hif4(player_ap, tuple(activation.shape))
            player_out = torch.matmul(player_a, player_w.transpose(-1, -2))
            bm, pm = mse(reference, baseline_out), mse(reference, player_out)
            case_score = score(bm, pm)
            all_scores.append(case_score)
            print(
                f"linear g={group_idx} t={test_idx}: baseline={bm:.8g} "
                f"player={pm:.8g} score={case_score:.6f}"
            )
    return all_scores, time.perf_counter() - started


def evaluate_attention(solution, groups: list) -> tuple[list[float], float]:
    all_scores: list[float] = []
    started = time.perf_counter()
    for group_idx, group in enumerate(groups):
        qh, kvh, dim = group["q_num_heads"], group["kv_num_heads"], group["head_dim"]
        states = solution.hif4_calibration_attention(group["calib"], qh, kvh, dim)
        for test_idx, sample in enumerate(group["test"]):
            originals: dict[str, torch.Tensor] = {}
            baselines: dict[str, torch.Tensor] = {}
            players: dict[str, torch.Tensor] = {}
            for role in ("q", "k", "v"):
                quant, scale = sample[role]
                original = dequantize_nvfp4(quant, scale).float()
                originals[role] = original
                baselines[role] = dequantize_hif4(
                    quantize_to_hif4(original), tuple(original.shape)
                )
                fn = getattr(solution, f"hif4_dynamic_quantize_{role}")
                heads = qh if role == "q" else kvh
                params = fn(quant, scale, heads, dim, states[f"{role}_state"])
                players[role] = dequantize_hif4(params, tuple(original.shape))

            reference = gqa_attention(
                originals["q"], originals["k"], originals["v"], qh, kvh, dim
            )
            baseline_out = gqa_attention(
                baselines["q"], baselines["k"], baselines["v"], qh, kvh, dim
            )
            player_out = gqa_attention(
                players["q"], players["k"], players["v"], qh, kvh, dim
            )
            bm, pm = mse(reference, baseline_out), mse(reference, player_out)
            case_score = score(bm, pm)
            all_scores.append(case_score)
            print(
                f"attn   g={group_idx} t={test_idx}: baseline={bm:.8g} "
                f"player={pm:.8g} score={case_score:.6f}"
            )
    return all_scores, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution/solution.py")
    parser.add_argument(
        "--datasets-dir", default="local_fitted_models_attention_feedback_v2",
        help="Active diagnostic proxy bank; use mini_sample explicitly for format checks only.",
    )
    args = parser.parse_args()
    solution = load_solution(Path(args.solution).resolve())
    data_dir = Path(args.datasets_dir)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    print(f"dataset={data_dir.resolve()} schema={manifest.get('schema', 'unversioned')}")
    if not manifest.get("score_prediction_allowed", False):
        print("WARNING: diagnostic data only; do not extrapolate online points or runtime.")
    linear = torch.load(data_dir / "linear.pt", map_location="cpu", weights_only=False)
    attention = torch.load(data_dir / "attn.pt", map_location="cpu", weights_only=False)
    linear_scores, linear_time = evaluate_linear(solution, linear)
    attention_scores, attention_time = evaluate_attention(solution, attention)
    scores = linear_scores + attention_scores
    print(
        f"summary: cases={len(scores)} score_sum={sum(scores):.6f} "
        f"score_mean={sum(scores) / len(scores):.6f} "
        f"linear_s={linear_time:.3f} attention_s={attention_time:.3f}"
    )


if __name__ == "__main__":
    main()
