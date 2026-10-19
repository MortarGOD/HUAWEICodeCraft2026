#!/usr/bin/env python3
"""Use current-test Q<->K predictions for direct softmax candidate routing."""

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


def _merge_blocks(params: list[dict], best: torch.Tensor):
    rows, blocks = params[0]["scale_factor"].shape[:2]
    selected = best.reshape(-1)
    result = {}
    for name in params[0]:
        stacked = torch.stack([candidate[name] for candidate in params])
        tail = stacked.shape[3:]
        index = selected.reshape((1, 1, blocks) + (1,) * len(tail))
        index = index.expand((1, rows, blocks) + tail)
        result[name] = torch.gather(stacked, 0, index).squeeze(0)
    return result


def install(solution, roles: str) -> None:
    original_q = solution.hif4_dynamic_quantize_q
    original_k = solution.hif4_dynamic_quantize_k

    def bank(values, parent):
        return [parent] + [
            solution._quantize_attention_values(
                values,
                solution._ATTENTION_QK_OFFSETS,
                solution._ATTENTION_ENERGY_PENALTY,
                (threshold,),
            )
            for threshold in (0.5, 0.475, 0.45, 0.4375, 0.425)
        ]

    def quantize_q(q_quant, q_scale, q_heads, dim, state):
        parent = original_q(q_quant, q_scale, q_heads, dim, state)
        raw = dequantize_nvfp4(q_quant, q_scale).to(torch.float32)
        policy = solution._attention_policy(state, int(raw.shape[0]))
        if "q" not in roles or policy != 0 or not bool(state["predictor_enabled"]):
            return parent
        values = solution._transform_attention_qk(raw, q_heads, dim, state)
        kv_heads = int(state["predicted_k_heads"])
        replicas = q_heads // kv_heads
        predictor_q = values.reshape(-1, kv_heads, replicas, dim).mean(2)
        predicted_k = torch.einsum(
            "tkd,kde->tke",
            predictor_q - state["predictor_center"].to(
                values.device, torch.float32
            ).unsqueeze(0),
            state["predictor_map"].to(values.device, torch.float32),
        ).repeat_interleave(replicas, dim=1)
        full = values.reshape(-1, q_heads, dim).transpose(0, 1)
        peer = predicted_k.transpose(0, 1)
        target = torch.softmax(
            torch.einsum("htd,hsd->hts", full, peer) / (dim ** 0.5),
            dim=-1,
        )
        candidates = bank(values, parent)
        scores = []
        for params in candidates:
            reconstructed = solution._dequantize_linear_params(params).reshape(
                -1, q_heads, dim
            ).transpose(0, 1)
            approximate = torch.softmax(
                torch.einsum("htd,hsd->hts", reconstructed, peer)
                / (dim ** 0.5),
                dim=-1,
            )
            if "t" in roles:
                scores.append((target - approximate).square().sum(1))
            else:
                scores.append((target - approximate).square().sum((1, 2)))
        best = torch.stack(scores).argmin(0)
        if "t" in roles:
            return _merge_token_heads(candidates, best.transpose(0, 1), dim)
        return _merge_heads(candidates, best, dim)

    def quantize_k(k_quant, k_scale, kv_heads, dim, state):
        parent = original_k(k_quant, k_scale, kv_heads, dim, state)
        raw = dequantize_nvfp4(k_quant, k_scale).to(torch.float32)
        policy = solution._attention_policy(state, int(raw.shape[0]))
        if "k" not in roles or policy != 0 or not bool(state["predictor_enabled"]):
            return parent
        values = solution._transform_attention_qk(raw, kv_heads, dim, state)
        values.reshape(-1, kv_heads, dim).sub_(
            state["center"].to(values.device).unsqueeze(0)
        )
        predicted_q = torch.einsum(
            "tkd,kde->tke",
            values.reshape(-1, kv_heads, dim),
            state["predictor_map"].to(values.device, torch.float32),
        )
        full = values.reshape(-1, kv_heads, dim)
        peer = predicted_q.transpose(0, 1)
        target = torch.softmax(
            torch.einsum("htd,shd->hts", peer, full) / (dim ** 0.5),
            dim=-1,
        )
        candidates = bank(values, parent)
        if "b" in roles:
            blocks_per_head = dim // 64
            reconstructed_bank = torch.stack([
                solution._dequantize_linear_params(params).reshape(
                    -1, kv_heads, blocks_per_head, 64
                )
                for params in candidates
            ])
            current = reconstructed_bank[0].clone()
            choices = torch.zeros(
                (kv_heads, blocks_per_head), dtype=torch.int64
            )
            for block in range(blocks_per_head):
                block_scores = []
                for candidate_index in range(len(candidates)):
                    proposal = current.clone()
                    proposal[:, :, block] = reconstructed_bank[
                        candidate_index, :, :, block
                    ]
                    approximate = torch.softmax(
                        torch.einsum(
                            "htd,shd->hts", peer,
                            proposal.reshape(-1, kv_heads, dim),
                        ) / (dim ** 0.5),
                        dim=-1,
                    )
                    block_scores.append(
                        (target - approximate).square().sum((1, 2))
                    )
                best = torch.stack(block_scores).argmin(0)
                choices[:, block] = best
                source = reconstructed_bank[:, :, :, block].permute(2, 0, 1, 3)
                gather_index = best.reshape(kv_heads, 1, 1, 1).expand(
                    kv_heads, 1, current.shape[0], 64
                )
                current[:, :, block] = torch.gather(
                    source, 1, gather_index
                ).squeeze(1).transpose(0, 1)
            return _merge_blocks(candidates, choices)
        scores = []
        for params in candidates:
            reconstructed = solution._dequantize_linear_params(params).reshape(
                -1, kv_heads, dim
            )
            approximate = torch.softmax(
                torch.einsum("htd,shd->hts", peer, reconstructed)
                / (dim ** 0.5),
                dim=-1,
            )
            if "t" in roles:
                scores.append((target - approximate).square().sum(1))
            else:
                scores.append((target - approximate).square().sum((1, 2)))
        best = torch.stack(scores).argmin(0)
        if "t" in roles:
            return _merge_token_heads(candidates, best.transpose(0, 1), dim)
        return _merge_heads(candidates, best, dim)

    solution.hif4_dynamic_quantize_q = quantize_q
    solution.hif4_dynamic_quantize_k = quantize_k


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--version", type=int, default=296)
    parser.add_argument("--roles", nargs="+", default=("", "q", "k", "qk"))
    args = parser.parse_args()
    for dataset in args.datasets:
        groups = torch.load(
            Path(dataset) / "attn.pt", map_location="cpu", weights_only=False
        )
        for roles in args.roles:
            solution = load_solution(
                ROOT / "solution" / f"solution_v{args.version}.py"
            )
            install(solution, roles)
            with contextlib.redirect_stdout(io.StringIO()):
                scores, elapsed = evaluate_attention(solution, groups)
            print(
                dataset, roles or "current", f"score={sum(scores):.6f}",
                f"seconds={elapsed:.2f}", flush=True,
            )


if __name__ == "__main__":
    main()
