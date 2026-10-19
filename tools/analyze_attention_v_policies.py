#!/usr/bin/env python3
"""Measure V rounding and token-mean error-shaping candidates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_local import gqa_attention, mse, score
from hif4_quant import dequantize_hif4, dequantize_nvfp4, quantize_to_hif4


def load_solution(version: int):
    path = ROOT / "solution" / f"solution_v{version}.py"
    spec = importlib.util.spec_from_file_location(f"vpolicy_v{version}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--version", type=int, default=291)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    solution = load_solution(args.version)
    report = {}
    for dataset in args.datasets:
        groups = torch.load(
            Path(dataset) / "attn.pt", map_location="cpu", weights_only=False
        )
        records = []
        totals = {name: 0.0 for name in (
            "current", "round375", "round625", "mean05", "mean10", "mean15"
        )}
        oracle = 0.0
        layout_totals = [{name: 0.0 for name in totals} for _ in range(5)]
        for group_index, group in enumerate(groups):
            q_heads = int(group["q_num_heads"])
            kv_heads = int(group["kv_num_heads"])
            head_dim = int(group["head_dim"])
            states = solution.hif4_calibration_attention(
                group["calib"], q_heads, kv_heads, head_dim
            )
            group_scores = {name: [] for name in totals}
            for layout, sample in enumerate(group["test"]):
                originals = {
                    role: dequantize_nvfp4(*sample[role]).float()
                    for role in ("q", "k", "v")
                }
                baselines = {
                    role: dequantize_hif4(
                        quantize_to_hif4(original), tuple(original.shape)
                    ).float()
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
                q_params = solution.hif4_dynamic_quantize_q(
                    *sample["q"], q_heads, head_dim, states["q_state"]
                )
                k_params = solution.hif4_dynamic_quantize_k(
                    *sample["k"], kv_heads, head_dim, states["k_state"]
                )
                quantized_q = solution._dequantize_linear_params(q_params)
                quantized_k = solution._dequantize_linear_params(k_params)
                value = originals["v"]
                current_params = solution._quantize_linear_values(
                    value, offsets=solution._ATTENTION_V_OFFSETS
                )
                current_v = solution._dequantize_linear_params(current_params)
                candidates = {
                    "current": current_v,
                    "round375": solution._dequantize_linear_params(
                        solution._quantize_linear_values(
                            value,
                            offsets=solution._ATTENTION_V_OFFSETS,
                            rounding_threshold=0.375,
                        )
                    ),
                    "round625": solution._dequantize_linear_params(
                        solution._quantize_linear_values(
                            value,
                            offsets=solution._ATTENTION_V_OFFSETS,
                            rounding_threshold=0.625,
                        )
                    ),
                }
                mean_error = (value - current_v).reshape(
                    value.shape[0], kv_heads, head_dim
                ).mean(0).reshape(1, -1)
                for name, alpha in (("mean05", 0.5), ("mean10", 1.0), ("mean15", 1.5)):
                    candidates[name] = solution._dequantize_linear_params(
                        solution._quantize_linear_values(
                            value + alpha * mean_error,
                            offsets=solution._ATTENTION_V_OFFSETS,
                        )
                    )
                scores = {}
                for name, quantized_v in candidates.items():
                    output = gqa_attention(
                        quantized_q, quantized_k, quantized_v,
                        q_heads, kv_heads, head_dim,
                    )
                    scores[name] = score(
                        baseline_mse, mse(reference, output)
                    )
                    totals[name] += scores[name]
                    layout_totals[layout][name] += scores[name]
                    group_scores[name].append(scores[name])
                oracle += max(scores.values())
            records.append({
                "group": group_index,
                "q_nmse": float(states["q_state"]["predictor_nmse"]),
                "k_nmse": float(states["k_state"]["predictor_nmse"]),
                "scores": group_scores,
            })
        report[dataset] = records
        print(dataset, "totals", totals, "oracle", oracle, flush=True)
        print(dataset, "layouts", layout_totals, flush=True)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
