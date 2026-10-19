#!/usr/bin/env python3
"""Select Q/K rounding with current input against stored calibration peer."""

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


def _sample_rows(values: torch.Tensor, limit: int) -> torch.Tensor:
    rows = int(values.shape[0])
    if rows <= limit:
        return values
    indices = torch.linspace(0, rows - 1, limit).round().to(torch.int64)
    return values.index_select(0, indices)


def _layout_index(state: dict, rows: int) -> int:
    layouts = state["layout_rows"].to(torch.float32).clamp_min(1).log()
    target = torch.tensor(float(rows)).log()
    return int((layouts - target).abs().argmin())


def _merge_heads(params: list[dict], best: torch.Tensor, head_dim: int):
    blocks_per_head = head_dim // 64
    blocks = int(best.numel()) * blocks_per_head
    selected = best.repeat_interleave(blocks_per_head)
    result = {}
    for name in params[0]:
        stacked = torch.stack([candidate[name] for candidate in params])
        tail = stacked.shape[3:]
        index = selected.reshape((1, 1, blocks) + (1,) * len(tail))
        index = index.expand((1, stacked.shape[1], blocks) + tail)
        result[name] = torch.gather(stacked, 0, index).squeeze(0)
    return result


def _merge_token_heads(params: list[dict], best: torch.Tensor, head_dim: int):
    blocks_per_head = head_dim // 64
    rows, heads = best.shape
    blocks = heads * blocks_per_head
    selected = best.repeat_interleave(blocks_per_head, dim=1)
    result = {}
    for name in params[0]:
        stacked = torch.stack([candidate[name] for candidate in params])
        tail = stacked.shape[3:]
        index = selected.reshape((1, rows, blocks) + (1,) * len(tail))
        index = index.expand((1, rows, blocks) + tail)
        result[name] = torch.gather(stacked, 0, index).squeeze(0)
    return result


def install(solution, roles: str, sample_rows: int) -> None:
    original_calibration = solution.hif4_calibration_attention
    original_q = solution.hif4_dynamic_quantize_q
    original_k = solution.hif4_dynamic_quantize_k

    def calibration(calib, q_heads, kv_heads, dim):
        states = original_calibration(calib, q_heads, kv_heads, dim)
        raw_q, transformed_q, raw_k, transformed_k = [], [], [], []
        for sample in calib:
            q = dequantize_nvfp4(*sample["q"]).to(torch.float32)
            k = dequantize_nvfp4(*sample["k"]).to(torch.float32)
            raw_q.append(_sample_rows(q, sample_rows).to(torch.bfloat16))
            raw_k.append(_sample_rows(k, sample_rows).to(torch.bfloat16))
            tq = solution._transform_attention_qk(
                q, q_heads, dim, states["q_state"]
            )
            tk = solution._transform_attention_qk(
                k, kv_heads, dim, states["k_state"]
            )
            transformed_q.append(
                _sample_rows(tq, sample_rows).to(torch.bfloat16)
            )
            transformed_k.append(
                _sample_rows(tk, sample_rows).to(torch.bfloat16)
            )
        states["q_state"]["probe_k_raw"] = raw_k
        states["q_state"]["probe_k_transformed"] = transformed_k
        states["k_state"]["probe_q_raw"] = raw_q
        states["k_state"]["probe_q_transformed"] = transformed_q
        return states

    def candidate_bank(values, parent):
        candidates = [parent] + [
            solution._quantize_attention_values(
                values,
                solution._ATTENTION_QK_OFFSETS,
                solution._ATTENTION_ENERGY_PENALTY,
                (threshold,),
            )
            for threshold in (0.5, 0.475, 0.45, 0.4375, 0.425)
        ]
        if "e" in roles:
            candidates.extend(
                solution._quantize_attention_values(
                    values,
                    solution._ATTENTION_QK_OFFSETS,
                    0.0,
                    (threshold,),
                )
                for threshold in (0.5, 0.475, 0.45, 0.4375, 0.425)
            )
        return candidates

    def quantize_q(q_quant, q_scale, q_heads, dim, state):
        parent = original_q(q_quant, q_scale, q_heads, dim, state)
        if "q" not in roles:
            return parent
        raw = dequantize_nvfp4(q_quant, q_scale).to(torch.float32)
        policy = solution._attention_policy(state, int(raw.shape[0]))
        values = raw if policy == 3 else solution._transform_attention_qk(
            raw, q_heads, dim, state
        )
        index = _layout_index(state, int(raw.shape[0]))
        key = "probe_k_raw" if policy == 3 else "probe_k_transformed"
        peer = state[key][index].to(values.device, torch.float32)
        kv_heads = int(peer.shape[1]) // dim
        replicas = q_heads // kv_heads
        peer = peer.reshape(-1, kv_heads, dim).repeat_interleave(replicas, 1)
        full = values.reshape(-1, q_heads, dim).transpose(0, 1)
        target = torch.softmax(
            torch.einsum("htd,hsd->hts", full, peer.transpose(0, 1))
            / (dim ** 0.5),
            dim=-1,
        )
        candidates = candidate_bank(values, parent)
        scores = []
        for params in candidates:
            reconstructed = solution._dequantize_linear_params(params).reshape_as(
                values
            ).reshape(-1, q_heads, dim).transpose(0, 1)
            approximate = torch.softmax(
                torch.einsum(
                    "htd,hsd->hts", reconstructed, peer.transpose(0, 1)
                ) / (dim ** 0.5),
                dim=-1,
            )
            scores.append((target - approximate).square().sum((1, 2)))
        return _merge_heads(candidates, torch.stack(scores).argmin(0), dim)

    def quantize_k(k_quant, k_scale, kv_heads, dim, state):
        parent = original_k(k_quant, k_scale, kv_heads, dim, state)
        if "k" not in roles:
            return parent
        raw = dequantize_nvfp4(k_quant, k_scale).to(torch.float32)
        policy = solution._attention_policy(state, int(raw.shape[0]))
        values = raw if policy == 3 else solution._transform_attention_qk(
            raw, kv_heads, dim, state
        )
        if policy != 3:
            values.reshape(-1, kv_heads, dim).sub_(
                state["center"].to(values.device).unsqueeze(0)
            )
        index = _layout_index(state, int(raw.shape[0]))
        key = "probe_q_raw" if policy == 3 else "probe_q_transformed"
        if "a" in roles:
            peer = torch.cat(state[key]).to(values.device, torch.float32)
        else:
            peer = state[key][index].to(values.device, torch.float32)
        q_heads = int(peer.shape[1]) // dim
        replicas = q_heads // kv_heads
        peer = peer.reshape(-1, kv_heads, replicas, dim)
        full = values.reshape(-1, kv_heads, dim)
        target = torch.softmax(
            torch.einsum("thrd,shd->hrts", peer, full) / (dim ** 0.5),
            dim=-1,
        )
        candidates = candidate_bank(values, parent)
        scores = []
        for params in candidates:
            reconstructed = solution._dequantize_linear_params(params).reshape(
                -1, kv_heads, dim
            )
            approximate = torch.softmax(
                torch.einsum("thrd,shd->hrts", peer, reconstructed)
                / (dim ** 0.5),
                dim=-1,
            )
            if "t" in roles:
                scores.append((target - approximate).square().sum((1, 2)))
            else:
                scores.append((target - approximate).square().sum((1, 2, 3)))
        best = torch.stack(scores).argmin(0)
        if "t" in roles:
            return _merge_token_heads(candidates, best.transpose(0, 1), dim)
        return _merge_heads(candidates, best, dim)

    solution.hif4_calibration_attention = calibration
    solution.hif4_dynamic_quantize_q = quantize_q
    solution.hif4_dynamic_quantize_k = quantize_k


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--version", type=int, default=296)
    parser.add_argument("--roles", nargs="+", default=("", "q", "k", "qk"))
    parser.add_argument("--sample-rows", type=int, default=128)
    args = parser.parse_args()
    for dataset in args.datasets:
        groups = torch.load(
            Path(dataset) / "attn.pt", map_location="cpu", weights_only=False
        )
        for roles in args.roles:
            solution = load_solution(
                ROOT / "solution" / f"solution_v{args.version}.py"
            )
            install(solution, roles, args.sample_rows)
            with contextlib.redirect_stdout(io.StringIO()):
                scores, elapsed = evaluate_attention(solution, groups)
            print(
                dataset, roles or "current", f"score={sum(scores):.6f}",
                f"seconds={elapsed:.2f}", flush=True,
            )


if __name__ == "__main__":
    main()
