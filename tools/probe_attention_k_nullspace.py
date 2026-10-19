#!/usr/bin/env python3
"""Probe exact K token-shift nullspace choices on fitted Attention sets."""

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
from hif4_quant import dequantize_nvfp4


def _shift(value: torch.Tensor, heads: int, dim: int, mode: str) -> torch.Tensor:
    shaped = value.reshape(-1, heads, dim)
    if mode == "mean":
        center = shaped.mean(0)
    elif mode == "median":
        center = shaped.median(0).values
    elif mode == "midrange":
        center = 0.5 * (shaped.amin(0) + shaped.amax(0))
    elif mode == "trimmed":
        rows = int(shaped.shape[0])
        trim = max(rows // 8, 1)
        if rows <= 2 * trim:
            center = shaped.mean(0)
        else:
            center = shaped.sort(0).values[trim:-trim].mean(0)
    else:
        raise ValueError(mode)
    return (shaped - center.unsqueeze(0)).reshape_as(value)


def install(solution, mode: str) -> None:
    original_calibration = solution.hif4_calibration_attention
    original_quantize_q = solution.hif4_dynamic_quantize_q
    q_cache = {}

    def calibration(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
        states = original_calibration(
            calib_qkv_list, q_num_heads, kv_num_heads, head_dim
        )
        replicas = q_num_heads // kv_num_heads
        raw_covariances = []
        transformed_covariances = []
        for sample in calib_qkv_list:
            raw = dequantize_nvfp4(*sample["q"]).to(torch.float32)
            transformed = solution._transform_attention_qk(
                raw.clone(), q_num_heads, head_dim, states["q_state"]
            )
            for values, destination in (
                (raw, raw_covariances),
                (transformed, transformed_covariances),
            ):
                grouped = values.reshape(
                    -1, kv_num_heads, replicas, head_dim
                )
                destination.append(
                    torch.einsum("thrd,thre->hde", grouped, grouped)
                )
        states["k_state"]["probe_q_cov_raw"] = torch.stack(raw_covariances)
        states["k_state"]["probe_q_cov_transformed"] = torch.stack(
            transformed_covariances
        )
        return states

    solution.hif4_calibration_attention = calibration

    def quantize_q(q_quant, q_scale, q_num_heads, head_dim, q_state):
        params = original_quantize_q(
            q_quant, q_scale, q_num_heads, head_dim, q_state
        )
        if mode in ("scorebankactual", "actualroundbank", "actualroundsoftmax"):
            raw = dequantize_nvfp4(q_quant, q_scale).to(torch.float32)
            policy = solution._attention_policy(q_state, int(raw.shape[0]))
            full = raw if policy == 3 else solution._transform_attention_qk(
                raw, q_num_heads, head_dim, q_state
            )
            q_cache.clear()
            q_cache.update({
                "full": full.reshape(-1, q_num_heads, head_dim),
                "quantized": solution._dequantize_linear_params(params).reshape(
                    -1, q_num_heads, head_dim
                ),
                "rows": int(raw.shape[0]),
            })
        return params

    solution.hif4_dynamic_quantize_q = quantize_q

    def quantize_value(value, policy, rounding, kv_num_heads, head_dim, k_state):
        if policy in (1, 2, 3):
            return solution._quantize_attention_values(
                value,
                solution._ATTENTION_QK_OFFSETS,
                solution._ATTENTION_ENERGY_PENALTY,
                rounding,
            )
        if bool(k_state["predictor_enabled"]):
            predictor_map = k_state["predictor_map"].to(
                value.device, torch.float32
            )
            predicted_q = torch.einsum(
                "tkd,kde->tke",
                value.reshape(-1, kv_num_heads, head_dim),
                predictor_map,
            )
            metric = solution._attention_block_metric(
                predicted_q, kv_num_heads, head_dim
            )
            return solution._quantize_attention_metric(
                value,
                solution._ATTENTION_QK_OFFSETS,
                kv_num_heads,
                head_dim,
                metric,
                1.0,
                solution._ATTENTION_ENERGY_PENALTY,
            )
        return solution._quantize_attention_values(
            value,
            solution._ATTENTION_QK_OFFSETS,
            solution._ATTENTION_ENERGY_PENALTY,
            rounding,
        )

    def quantize_k(k_quant, k_scale, kv_num_heads, head_dim, k_state):
        value = dequantize_nvfp4(k_quant, k_scale).to(torch.float32)
        policy = solution._attention_policy(k_state, int(value.shape[0]))
        rounding = solution._attention_rounding_thresholds(
            k_state, int(value.shape[0])
        )
        if policy != 3:
            value = solution._transform_attention_qk(
                value, kv_num_heads, head_dim, k_state
            )
        shaped = value.reshape(-1, kv_num_heads, head_dim)
        if policy == 3:
            current_center = shaped.new_zeros((kv_num_heads, head_dim))
        else:
            current_center = k_state["center"].to(value.device)
        if mode in ("current", "actualroundbank", "actualroundsoftmax"):
            candidates = [value - current_center.unsqueeze(0).reshape(1, -1)]
        elif mode in ("bank", "scorebank", "scorebankactual"):
            test_center = shaped.mean(0)
            candidates = [
                value - (
                    current_center + alpha * (test_center - current_center)
                ).unsqueeze(0).reshape(1, -1)
                for alpha in (
                    (-0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25, 0.5, 1.0)
                    if mode in ("scorebank", "scorebankactual")
                    else (0.0, 0.125, 0.25, 0.5, 1.0)
                )
            ]
        elif mode.startswith("blend"):
            alpha = float(mode.removeprefix("blend"))
            test_center = shaped.mean(0)
            center = current_center + alpha * (test_center - current_center)
            candidates = [value - center.unsqueeze(0).reshape(1, -1)]
        else:
            candidates = [_shift(value, kv_num_heads, head_dim, mode)]
        if mode in ("actualroundbank", "actualroundsoftmax"):
            target = candidates[0]
            candidates = [target] * 6
            params = [
                quantize_value(
                    target, policy, rounding, kv_num_heads, head_dim, k_state
                )
            ] + [
                solution._quantize_attention_values(
                    target,
                    solution._ATTENTION_QK_OFFSETS,
                    solution._ATTENTION_ENERGY_PENALTY,
                    (threshold,),
                )
                for threshold in (0.5, 0.475, 0.45, 0.4375, 0.425)
            ]
        else:
            params = [
                quantize_value(
                    candidate, policy, rounding, kv_num_heads, head_dim, k_state
                )
                for candidate in candidates
            ]
        if len(params) == 1:
            return params[0]
        blocks = int(value.shape[-1]) // 64
        residuals = []
        for candidate, candidate_params in zip(candidates, params):
            reconstructed = solution._dequantize_linear_params(
                candidate_params
            ).reshape_as(candidate)
            residual = candidate - reconstructed
            if mode == "scorebank":
                rows = k_state["layout_rows"].to(torch.float32).clamp_min(1).log()
                index = int(
                    (rows - torch.tensor(float(value.shape[0])).log()).abs().argmin()
                )
                covariance = k_state[
                    "probe_q_cov_raw" if policy == 3 else "probe_q_cov_transformed"
                ][index].to(value.device, torch.float32)
                error = residual.reshape(-1, kv_num_heads, head_dim)
                residuals.append(
                    torch.einsum("thd,hde,the->h", error, covariance, error)
                )
            elif (
                mode in (
                    "scorebankactual", "actualroundbank", "actualroundsoftmax"
                )
                and q_cache.get("rows") == int(value.shape[0])
            ):
                full_k = candidate.reshape(-1, kv_num_heads, head_dim)
                quantized_k = reconstructed.reshape(
                    -1, kv_num_heads, head_dim
                )
                full_q = q_cache["full"].reshape(
                    -1, kv_num_heads,
                    q_cache["full"].shape[1] // kv_num_heads,
                    head_dim,
                )
                quantized_q = q_cache["quantized"].reshape_as(full_q)
                full_k = full_k - full_k.mean(0, keepdim=True)
                quantized_k = quantized_k - quantized_k.mean(0, keepdim=True)
                target = torch.einsum("thrd,shd->hrts", full_q, full_k)
                approximate = torch.einsum(
                    "thrd,shd->hrts", quantized_q, quantized_k
                )
                if mode == "actualroundsoftmax":
                    target = torch.softmax(target / (head_dim ** 0.5), dim=-1)
                    approximate = torch.softmax(
                        approximate / (head_dim ** 0.5), dim=-1
                    )
                residuals.append((target - approximate).square().sum((1, 2, 3)))
            else:
                residuals.append(
                    residual.square().reshape(-1, blocks, 64).sum((0, 2))
                )
        best = torch.stack(residuals).argmin(0)
        result = {}
        for name in params[0]:
            stacked = torch.stack([candidate[name] for candidate in params])
            tail = stacked.shape[3:]
            if mode in (
                "scorebank", "scorebankactual", "actualroundbank",
                "actualroundsoftmax",
            ):
                head_blocks = head_dim // 64
                selected = best.repeat_interleave(head_blocks)
            else:
                selected = best
            index = selected.reshape((1, 1, blocks) + (1,) * len(tail))
            index = index.expand((1, stacked.shape[1], blocks) + tail)
            result[name] = torch.gather(stacked, 0, index).squeeze(0)
        return result

    solution.hif4_dynamic_quantize_k = quantize_k


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--version", type=int, default=296)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=("current", "mean", "median", "midrange", "trimmed"),
    )
    args = parser.parse_args()
    for dataset in args.datasets:
        groups = torch.load(
            Path(dataset) / "attn.pt", map_location="cpu", weights_only=False
        )
        mode_scores = {}
        for mode in args.modes:
            solution = load_solution(
                ROOT / "solution" / f"solution_v{args.version}.py"
            )
            install(solution, mode)
            with contextlib.redirect_stdout(io.StringIO()):
                scores, elapsed = evaluate_attention(solution, groups)
            mode_scores[mode] = scores
            print(
                dataset,
                mode,
                f"score={sum(scores):.6f}",
                f"seconds={elapsed:.2f}",
                flush=True,
            )
        current = mode_scores.get("current")
        if current is not None:
            oracle = sum(max(values) for values in zip(*mode_scores.values()))
            print(
                dataset,
                f"oracle={oracle:.6f}",
                f"delta={oracle - sum(current):.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
