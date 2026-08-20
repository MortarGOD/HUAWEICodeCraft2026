"""Local evaluation tool based on datasets_0818 (not mini_sample).

Usage:
    /opt/miniconda3/envs/codecraft/bin/python local_eval.py \
        --solution solution/solution_v8.py \
        --linear-groups 0,1,2,3 \
        --attn-groups 0,1,2,3,4

The tool reports Linear and Attention MSE separately. It does not estimate
online runtime; the local CPU differs from the 96-core online judge.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent


def load_module(path: str):
    spec = importlib.util.spec_from_file_location("candidate_solution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load solution from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dequant_hif4(params, shape):
    return (
        params["sign"].to(torch.float32)
        * params["mant"].to(torch.float32)
        * params["scale_lv3"].to(torch.float32)
        * params["scale_lv2"].to(torch.float32)
        * params["scale_factor"].to(torch.float32)
    ).reshape(shape)


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a.float() - b.float()) ** 2).mean().item())


def run_attention(q, k, v, q_num_heads, kv_num_heads, head_dim):
    seq = int(q.shape[0])
    q = q.reshape(seq, q_num_heads, head_dim)
    k = k.reshape(seq, kv_num_heads, head_dim)
    v = v.reshape(seq, kv_num_heads, head_dim)
    ratio = q_num_heads // kv_num_heads
    k = k.repeat_interleave(ratio, dim=1)
    v = v.repeat_interleave(ratio, dim=1)
    scores = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
    probs = torch.softmax(scores, dim=-1)
    return probs @ v


def eval_linear(module, linear_data, group_ids):
    total = 0.0
    count = 0
    for gi in group_ids:
        group = linear_data[gi]
        w_quant, w_scale = group["weight"]
        weight = module.dequantize_nvfp4(w_quant, w_scale).float()
        result = module.hif4_calibration_and_quantize_weight(
            w_quant, w_scale, group["calib_activation_list"]
        )
        weight_hat = dequant_hif4(result["weight_params"], weight.shape)
        for a_quant, a_scale in group["test_activation_list"]:
            activation = module.dequantize_nvfp4(a_quant, a_scale).float()
            act_hat = dequant_hif4(
                module.hif4_dynamic_quantize_activation(
                    a_quant, a_scale, result["activation_state"]
                ),
                activation.shape,
            )
            total += mse(activation @ weight.t(), act_hat @ weight_hat.t())
            count += 1
    return total, count


def eval_attention(module, attn_data, group_ids):
    total = 0.0
    count = 0
    for gi in group_ids:
        group = attn_data[gi]
        qh = int(group["q_num_heads"])
        kvh = int(group["kv_num_heads"])
        hd = int(group["head_dim"])
        states = module.hif4_calibration_attention(group["calib"], qh, kvh, hd)
        for sample in group["test"]:
            outs = {}
            for role, func, state, nh in (
                ("q", module.hif4_dynamic_quantize_q, states["q_state"], qh),
                ("k", module.hif4_dynamic_quantize_k, states["k_state"], kvh),
                ("v", module.hif4_dynamic_quantize_v, states["v_state"], kvh),
            ):
                quant, scale = sample[role]
                value = module.dequantize_nvfp4(quant, scale).float()
                outs[role] = dequant_hif4(func(quant, scale, nh, hd, state), value.shape)

            q0 = module.dequantize_nvfp4(sample["q"][0], sample["q"][1]).float()
            k0 = module.dequantize_nvfp4(sample["k"][0], sample["k"][1]).float()
            v0 = module.dequantize_nvfp4(sample["v"][0], sample["v"][1]).float()
            total += mse(
                run_attention(q0, k0, v0, qh, kvh, hd),
                run_attention(outs["q"], outs["k"], outs["v"], qh, kvh, hd),
            )
            count += 1
    return total, count


def parse_ids(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip() != ""]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True)
    parser.add_argument("--linear-groups", default="0,1,2,3")
    parser.add_argument("--attn-groups", default="0,1,2,3,4")
    args = parser.parse_args()

    module = load_module(args.solution)
    linear_ids = parse_ids(args.linear_groups)
    attn_ids = parse_ids(args.attn_groups)

    linear_data = torch.load(
        ROOT / "datasets_0818" / "linear.pt", weights_only=True, map_location="cpu"
    )
    attn_data = torch.load(
        ROOT / "datasets_0818" / "attn.pt", weights_only=True, map_location="cpu"
    )

    lin_total, lin_count = eval_linear(module, linear_data, linear_ids)
    attn_total, attn_count = eval_attention(module, attn_data, attn_ids)
    print(f"Linear  groups={linear_ids} cases={lin_count} total_mse={lin_total:.6f} mean_mse={lin_total / lin_count:.6f}")
    print(f"Attention groups={attn_ids} cases={attn_count} total_mse={attn_total:.6f} mean_mse={attn_total / attn_count:.6f}")


if __name__ == "__main__":
    main()
