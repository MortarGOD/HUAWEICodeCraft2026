#!/usr/bin/env python3
"""Record per-group transfer of the narrow multi-rounding Q/K bank."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_local import evaluate_attention, gqa_attention, load_solution, mse
from hif4_quant import dequantize_hif4, dequantize_nvfp4, quantize_to_hif4
from probe_attention_rounding import install_bank
from probe_attention_rounding_roles import install_role_bank


@torch.inference_mode()
def calibration_gains(base, bank, group, states) -> list[float]:
    q_heads = int(group["q_num_heads"])
    kv_heads = int(group["kv_num_heads"])
    head_dim = int(group["head_dim"])
    gains = []
    for sample in group["calib"]:
        originals = {
            role: dequantize_nvfp4(*sample[role]).float()
            for role in ("q", "k", "v")
        }
        reference = gqa_attention(
            originals["q"], originals["k"], originals["v"],
            q_heads, kv_heads, head_dim,
        )
        baselines = {
            role: dequantize_hif4(
                quantize_to_hif4(value), tuple(value.shape)
            ).float()
            for role, value in originals.items()
        }
        baseline = gqa_attention(
            baselines["q"], baselines["k"], baselines["v"],
            q_heads, kv_heads, head_dim,
        )
        outputs = []
        for solution in (base, bank):
            quantized = {}
            for role in ("q", "k", "v"):
                heads = q_heads if role == "q" else kv_heads
                params = getattr(solution, f"hif4_dynamic_quantize_{role}")(
                    *sample[role], heads, head_dim, states[f"{role}_state"]
                )
                quantized[role] = solution._dequantize_linear_params(params)
            outputs.append(gqa_attention(
                quantized["q"], quantized["k"], quantized["v"],
                q_heads, kv_heads, head_dim,
            ))
        denominator = max(mse(reference, baseline), 1.0e-30)
        gains.append((mse(reference, outputs[0]) - mse(reference, outputs[1])) / denominator)
    return gains


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--version", type=int, default=291)
    parser.add_argument("--thresholds", default="0.4375,0.5,0.5625")
    parser.add_argument("--roles", default="qkv")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    thresholds = tuple(float(x) for x in args.thresholds.split(",") if x)
    report = {}
    for dataset in args.datasets:
        groups = torch.load(Path(dataset) / "attn.pt", weights_only=False)
        base = load_solution(Path(f"solution/solution_v{args.version}.py").resolve())
        bank = load_solution(Path(f"solution/solution_v{args.version}.py").resolve())
        if args.roles == "qkv":
            install_bank(bank, thresholds)
        else:
            install_role_bank(bank, args.roles, thresholds)
        records = []
        for index, group in enumerate(groups):
            states = base.hif4_calibration_attention(
                group["calib"], int(group["q_num_heads"]),
                int(group["kv_num_heads"]), int(group["head_dim"]),
            )
            with contextlib.redirect_stdout(io.StringIO()):
                base_scores, _ = evaluate_attention(base, [group])
                bank_scores, _ = evaluate_attention(bank, [group])
            record = {
                "group": index,
                "q_nmse": float(states["q_state"]["predictor_nmse"]),
                "k_nmse": float(states["k_state"]["predictor_nmse"]),
                "hadamard": int(states["q_state"]["hadamard"]),
                "base": sum(base_scores),
                "bank": sum(bank_scores),
                "gain": sum(bank_scores) - sum(base_scores),
                "layout_gains": [b - a for a, b in zip(base_scores, bank_scores)],
                "calibration_layout_gains": calibration_gains(
                    base, bank, group, states
                ),
            }
            records.append(record)
        report[dataset] = records
        print(
            dataset,
            f"base={sum(x['base'] for x in records):.6f}",
            f"bank={sum(x['bank'] for x in records):.6f}",
            f"gain={sum(x['gain'] for x in records):.6f}",
            flush=True,
        )
    args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
