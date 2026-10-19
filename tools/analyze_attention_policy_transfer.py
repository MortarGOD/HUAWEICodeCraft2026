#!/usr/bin/env python3
"""Measure calibration features against current-vs-raw Q/K test policy."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hif4_quant import dequantize_hif4, dequantize_nvfp4, quantize_to_hif4
from evaluate_local import gqa_attention, mse, score


def load_solution(version: int):
    path = ROOT / "solution" / f"solution_v{version}.py"
    spec = importlib.util.spec_from_file_location(f"policy_v{version}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def analyze(solution, groups: list[dict]) -> list[dict]:
    records = []
    for group_index, group in enumerate(groups):
        q_heads = int(group["q_num_heads"])
        kv_heads = int(group["kv_num_heads"])
        head_dim = int(group["head_dim"])
        states = solution.hif4_calibration_attention(
            group["calib"], q_heads, kv_heads, head_dim
        )
        policy_scores = {0: [], 2: [], 3: []}
        mixed_policy_scores = {
            f"{q_policy}{k_policy}": []
            for q_policy in (0, 2, 3)
            for k_policy in (0, 2, 3)
        }
        test_layout_indices = []
        for sample in group["test"]:
            originals = {
                role: dequantize_nvfp4(*sample[role]).to(torch.float32)
                for role in ("q", "k", "v")
            }
            baselines = {
                role: dequantize_hif4(
                    quantize_to_hif4(original), tuple(original.shape)
                ).to(torch.float32)
                for role, original in originals.items()
            }
            reference = gqa_attention(
                originals["q"], originals["k"], originals["v"],
                q_heads, kv_heads, head_dim,
            )
            baseline = gqa_attention(
                baselines["q"], baselines["k"], baselines["v"],
                q_heads, kv_heads, head_dim,
            )
            baseline_mse = mse(reference, baseline)
            layout_rows = states["q_state"]["layout_rows"].float().clamp_min(1.0)
            test_rows = max(int(originals["q"].shape[0]), 1)
            test_layout_indices.append(int(
                (layout_rows.log() - torch.tensor(float(test_rows)).log())
                .abs().argmin()
            ))
            v_params = solution.hif4_dynamic_quantize_v(
                *sample["v"], kv_heads, head_dim, states["v_state"]
            )
            quantized_v = solution._dequantize_linear_params(
                v_params
            ).reshape_as(originals["v"])
            quantized_policies = {}
            for policy in (0, 2, 3):
                q_state = dict(states["q_state"])
                k_state = dict(states["k_state"])
                q_state["forced_policy"] = policy
                k_state["forced_policy"] = policy
                q_params = solution.hif4_dynamic_quantize_q(
                    *sample["q"], q_heads, head_dim, q_state
                )
                k_params = solution.hif4_dynamic_quantize_k(
                    *sample["k"], kv_heads, head_dim, k_state
                )
                quantized_q = solution._dequantize_linear_params(
                    q_params
                ).reshape_as(originals["q"])
                quantized_k = solution._dequantize_linear_params(
                    k_params
                ).reshape_as(originals["k"])
                quantized_policies[policy] = (quantized_q, quantized_k)
                output = gqa_attention(
                    quantized_q, quantized_k, quantized_v,
                    q_heads, kv_heads, head_dim,
                )
                policy_scores[policy].append(
                    score(baseline_mse, mse(reference, output))
                )
            for q_policy in (0, 2, 3):
                for k_policy in (0, 2, 3):
                    quantized_q = quantized_policies[q_policy][0]
                    quantized_k = quantized_policies[k_policy][1]
                    output = gqa_attention(
                        quantized_q, quantized_k, quantized_v,
                        q_heads, kv_heads, head_dim,
                    )
                    mixed_policy_scores[f"{q_policy}{k_policy}"].append(
                        score(baseline_mse, mse(reference, output))
                    )
        q_state = states["q_state"]
        k_state = states["k_state"]
        records.append({
            "group": group_index,
            "q_nmse": float(q_state["predictor_nmse"]),
            "k_nmse": float(k_state["predictor_nmse"]),
            "hadamard": int(q_state["hadamard"]),
            "calib_raw_count": int((q_state["layout_policies"] == 3).sum()),
            "current": sum(policy_scores[0]),
            "transformed": sum(policy_scores[2]),
            "raw": sum(policy_scores[3]),
            "gain": sum(policy_scores[3]) - sum(policy_scores[0]),
            "policy_scores": policy_scores,
            "mixed_policy_scores": mixed_policy_scores,
            "test_layout_indices": test_layout_indices,
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--version", type=int, default=287)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    solution = load_solution(args.version)
    report = {}
    for name in args.datasets:
        path = Path(name)
        groups = torch.load(
            path / "attn.pt", map_location="cpu", weights_only=False
        )
        records = analyze(solution, groups)
        report[name] = records
        print(
            name,
            f"groups={len(records)}",
            f"current={sum(item['current'] for item in records):.6f}",
            f"transformed={sum(item['transformed'] for item in records):.6f}",
            f"raw={sum(item['raw'] for item in records):.6f}",
            f"gain={sum(item['gain'] for item in records):.6f}",
            flush=True,
        )
        layout_totals = {
            policy: [
                sum(record["policy_scores"][policy][layout] for record in records)
                for layout in range(len(records[0]["policy_scores"][policy]))
            ]
            for policy in (0, 2, 3)
        }
        oracle = sum(
            max(record["policy_scores"][policy][layout] for policy in (0, 2, 3))
            for record in records
            for layout in range(len(record["policy_scores"][0]))
        )
        print(name, f"layouts={layout_totals} oracle={oracle:.6f}", flush=True)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
