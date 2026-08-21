"""Leakage-resistant local evaluation based on datasets_0818 source tensors.

Usage:
    /opt/miniconda3/envs/codecraft/bin/python local_eval.py \
        --solution solution/solution_v14.py \
        --linear-groups 0,1,2,3 \
        --attn-groups 0,1,2,3,4 \
        --calibrate-online

By default, test tensors are taken from a structurally identical, calibration-
feature-matched donor group so the copied 0818 calibration/test values cannot
leak. With ``--calibrate-online``, Linear deltas are calibrated from the
controlled v21-to-v22 online change. Attention deltas receive zero
extrapolation weight in the conservative estimate; mixed external results are
recorded separately. ``--leave-one-out`` instead preserves the original group
but excludes each test tensor's exact calibration twin before fitting state.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
DEFAULT_PROXY_BASELINE = ROOT / "solution" / "solution_v1.py"
DEFAULT_ONLINE_REFERENCE = ROOT / "solution" / "solution_v22.py"
DEFAULT_REFERENCE_ONLINE_SCORE = 16359.0
DEFAULT_REFERENCE_ONLINE_SECONDS = 197.0
DEFAULT_LINEAR_CALIBRATION_REFERENCE = ROOT / "solution" / "solution_v21.py"
DEFAULT_LINEAR_CALIBRATION_ONLINE_SCORE = 15678.0
DEFAULT_HOLDOUT_MANIFEST = ROOT / "datasets_holdout_v2" / "manifest.json"


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
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
    q = q.reshape(seq, q_num_heads, head_dim).transpose(0, 1)
    k = k.reshape(seq, kv_num_heads, head_dim).transpose(0, 1)
    v = v.reshape(seq, kv_num_heads, head_dim).transpose(0, 1)
    ratio = q_num_heads // kv_num_heads
    k = k.repeat_interleave(ratio, dim=0)
    v = v.repeat_interleave(ratio, dim=0)
    scores = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
    probs = torch.softmax(scores, dim=-1)
    return (probs @ v).transpose(0, 1).flatten(-2, -1)


def _tensor_pair_equal(left, right) -> bool:
    return torch.equal(left[0], right[0]) and torch.equal(left[1], right[1])


def _attention_sample_equal(left, right) -> bool:
    return all(
        _tensor_pair_equal(left[role], right[role])
        for role in ("q", "k", "v")
    )


def eval_linear(module, linear_data, group_ids, leave_one_out=False):
    errors = []
    started = time.perf_counter()
    for gi in group_ids:
        group = linear_data[gi]
        w_quant, w_scale = group["weight"]
        weight = module.dequantize_nvfp4(w_quant, w_scale).float()
        tests = group["test_activation_list"]
        shared_result = None
        if not leave_one_out:
            shared_result = module.hif4_calibration_and_quantize_weight(
                w_quant, w_scale, group["calib_activation_list"]
            )
        for test in tests:
            result = shared_result
            if leave_one_out:
                matches = [
                    index
                    for index, item in enumerate(group["calib_activation_list"])
                    if _tensor_pair_equal(test, item)
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Linear group {gi} test requires one exact twin; "
                        f"found {len(matches)}"
                    )
                calib = [
                    item
                    for index, item in enumerate(group["calib_activation_list"])
                    if index != matches[0]
                ]
                result = module.hif4_calibration_and_quantize_weight(
                    w_quant, w_scale, calib
                )
            weight_hat = dequant_hif4(result["weight_params"], weight.shape)
            a_quant, a_scale = test
            activation = module.dequantize_nvfp4(a_quant, a_scale).float()
            act_hat = dequant_hif4(
                module.hif4_dynamic_quantize_activation(
                    a_quant, a_scale, result["activation_state"]
                ),
                activation.shape,
            )
            errors.append(mse(activation @ weight.t(), act_hat @ weight_hat.t()))
    return errors, time.perf_counter() - started


def eval_attention(module, attn_data, group_ids, leave_one_out=False):
    errors = []
    started = time.perf_counter()
    for gi in group_ids:
        group = attn_data[gi]
        qh = int(group["q_num_heads"])
        kvh = int(group["kv_num_heads"])
        hd = int(group["head_dim"])
        shared_states = None
        if not leave_one_out:
            shared_states = module.hif4_calibration_attention(
                group["calib"], qh, kvh, hd
            )
        for sample in group["test"]:
            states = shared_states
            if leave_one_out:
                matches = [
                    index
                    for index, item in enumerate(group["calib"])
                    if _attention_sample_equal(sample, item)
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Attention group {gi} test requires one exact twin; "
                        f"found {len(matches)}"
                    )
                calib = [
                    item
                    for index, item in enumerate(group["calib"])
                    if index != matches[0]
                ]
                states = module.hif4_calibration_attention(calib, qh, kvh, hd)
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
            errors.append(mse(
                run_attention(q0, k0, v0, qh, kvh, hd),
                run_attention(outs["q"], outs["k"], outs["v"], qh, kvh, hd),
            ))
    return errors, time.perf_counter() - started


def parse_ids(text: str, group_count: int) -> list[int]:
    if text.strip().lower() == "all":
        return list(range(group_count))
    result = [int(x) for x in text.split(",") if x.strip() != ""]
    invalid = [idx for idx in result if idx < 0 or idx >= group_count]
    if invalid:
        raise ValueError(f"group ids out of range 0..{group_count - 1}: {invalid}")
    return result


def apply_holdout_manifest(linear_data, attn_data, manifest_path: str):
    """Replace copied tests with independent, structurally matched donors."""
    if not manifest_path:
        return linear_data, attn_data
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    def donor_index(mapping, index, count, kind):
        donor = int(mapping[str(index)])
        if donor < 0 or donor >= count or donor == index:
            raise ValueError(f"invalid {kind} donor {index}->{donor}")
        return donor

    linear_mapping = manifest["linear_donors"]
    remapped_linear = []
    for index, group in enumerate(linear_data):
        donor = linear_data[
            donor_index(linear_mapping, index, len(linear_data), "Linear")
        ]
        receiver_shapes = [
            tuple(item[0].shape) for item in group["test_activation_list"]
        ]
        donor_shapes = [
            tuple(item[0].shape) for item in donor["test_activation_list"]
        ]
        if receiver_shapes != donor_shapes:
            raise ValueError(f"Linear donor shape mismatch for group {index}")
        remapped = dict(group)
        remapped["test_activation_list"] = donor["test_activation_list"]
        remapped_linear.append(remapped)

    attention_mapping = manifest["attention_donors"]
    remapped_attention = []
    for index, group in enumerate(attn_data):
        donor = attn_data[
            donor_index(attention_mapping, index, len(attn_data), "Attention")
        ]
        receiver_signature = (
            int(group["q_num_heads"]),
            int(group["kv_num_heads"]),
            int(group["head_dim"]),
            [tuple(item["q"][0].shape) for item in group["test"]],
        )
        donor_signature = (
            int(donor["q_num_heads"]),
            int(donor["kv_num_heads"]),
            int(donor["head_dim"]),
            [tuple(item["q"][0].shape) for item in donor["test"]],
        )
        if receiver_signature != donor_signature:
            raise ValueError(f"Attention donor signature mismatch for group {index}")
        remapped = dict(group)
        remapped["test"] = donor["test"]
        remapped_attention.append(remapped)
    return remapped_linear, remapped_attention


def evaluate(
    module,
    linear_data,
    attn_data,
    linear_ids,
    attn_ids,
    leave_one_out=False,
):
    linear_errors, linear_seconds = eval_linear(
        module, linear_data, linear_ids, leave_one_out
    )
    attn_errors, attn_seconds = eval_attention(
        module, attn_data, attn_ids, leave_one_out
    )
    return {
        "linear_errors": linear_errors,
        "attn_errors": attn_errors,
        "linear_seconds": linear_seconds,
        "attn_seconds": attn_seconds,
        "seconds": linear_seconds + attn_seconds,
    }


def print_raw(label: str, result: dict, linear_ids: list[int], attn_ids: list[int]):
    linear_errors = result["linear_errors"]
    attn_errors = result["attn_errors"]
    linear_total = sum(linear_errors)
    attn_total = sum(attn_errors)
    print(
        f"{label} Linear: groups={linear_ids} cases={len(linear_errors)} "
        f"total_mse={linear_total:.6f} "
        f"mean_mse={linear_total / max(1, len(linear_errors)):.6f} "
        f"seconds={result['linear_seconds']:.3f}"
    )
    print(
        f"{label} Attention: groups={attn_ids} cases={len(attn_errors)} "
        f"total_mse={attn_total:.6f} "
        f"mean_mse={attn_total / max(1, len(attn_errors)):.6f} "
        f"seconds={result['attn_seconds']:.3f}"
    )
    print(f"{label} local_total_seconds={result['seconds']:.3f}")


def proxy_score(result: dict, baseline: dict) -> tuple[float, float, float]:
    def component(player_errors, baseline_errors):
        if len(player_errors) != len(baseline_errors):
            raise ValueError("candidate and proxy baseline case counts differ")
        return 100.0 * sum(
            (base - player) / base
            for player, base in zip(player_errors, baseline_errors)
            if base > 0.0
        )

    linear = component(result["linear_errors"], baseline["linear_errors"])
    attention = component(result["attn_errors"], baseline["attn_errors"])
    return linear, attention, linear + attention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True)
    parser.add_argument("--linear-groups", default="0,1,2,3")
    parser.add_argument("--attn-groups", default="0,1,2,3,4")
    parser.add_argument("--calibrate-online", action="store_true")
    parser.add_argument(
        "--leave-one-out",
        action="store_true",
        help="exclude each raw test tensor's exact calibration twin",
    )
    parser.add_argument(
        "--proxy-baseline", default=str(DEFAULT_PROXY_BASELINE)
    )
    parser.add_argument(
        "--online-reference", default=str(DEFAULT_ONLINE_REFERENCE)
    )
    parser.add_argument(
        "--reference-online-score",
        type=float,
        default=DEFAULT_REFERENCE_ONLINE_SCORE,
    )
    parser.add_argument(
        "--reference-online-seconds",
        type=float,
        default=DEFAULT_REFERENCE_ONLINE_SECONDS,
    )
    parser.add_argument(
        "--linear-calibration-reference",
        default=str(DEFAULT_LINEAR_CALIBRATION_REFERENCE),
    )
    parser.add_argument(
        "--linear-calibration-online-score",
        type=float,
        default=DEFAULT_LINEAR_CALIBRATION_ONLINE_SCORE,
    )
    parser.add_argument(
        "--holdout-manifest",
        default=str(DEFAULT_HOLDOUT_MANIFEST),
        help="cross-group donor manifest; pass an empty string for leaked raw tests",
    )
    args = parser.parse_args()

    linear_data = torch.load(
        ROOT / "datasets_0818" / "linear.pt", weights_only=True, map_location="cpu"
    )
    attn_data = torch.load(
        ROOT / "datasets_0818" / "attn.pt", weights_only=True, map_location="cpu"
    )
    if args.leave_one_out:
        print("Holdout mode: within-group exact-twin-excluded")
    elif args.holdout_manifest:
        linear_data, attn_data = apply_holdout_manifest(
            linear_data, attn_data, args.holdout_manifest
        )
        print(f"Holdout manifest: {args.holdout_manifest}")
    else:
        print("Holdout manifest: disabled (raw tests contain known leakage)")
    linear_ids = parse_ids(args.linear_groups, len(linear_data))
    attn_ids = parse_ids(args.attn_groups, len(attn_data))

    module = load_module(args.solution, "candidate_solution")
    candidate = evaluate(
        module,
        linear_data,
        attn_data,
        linear_ids,
        attn_ids,
        args.leave_one_out,
    )
    print_raw("Candidate", candidate, linear_ids, attn_ids)

    if not args.calibrate_online:
        return

    baseline_module = load_module(args.proxy_baseline, "proxy_baseline_solution")
    baseline = evaluate(
        baseline_module,
        linear_data,
        attn_data,
        linear_ids,
        attn_ids,
        args.leave_one_out,
    )
    if Path(args.solution).resolve() == Path(args.online_reference).resolve():
        reference = candidate
    else:
        reference_module = load_module(
            args.online_reference, "online_reference_solution"
        )
        reference = evaluate(
            reference_module,
            linear_data,
            attn_data,
            linear_ids,
            attn_ids,
            args.leave_one_out,
        )
    linear_calibration_module = load_module(
        args.linear_calibration_reference, "linear_calibration_solution"
    )
    linear_calibration = evaluate(
        linear_calibration_module,
        linear_data,
        attn_data,
        linear_ids,
        [],
        args.leave_one_out,
    )
    print_raw("Reference", reference, linear_ids, attn_ids)

    cand_lin, cand_attn, cand_proxy = proxy_score(candidate, baseline)
    ref_lin, ref_attn, ref_proxy = proxy_score(reference, baseline)
    if reference["seconds"] <= 0.0:
        raise RuntimeError("reference runtime is zero; cannot calibrate")
    calibration_lin, _, _ = proxy_score(
        linear_calibration,
        {
            "linear_errors": baseline["linear_errors"],
            "attn_errors": [],
        },
    )
    online_linear_delta = (
        args.reference_online_score - args.linear_calibration_online_score
    )
    proxy_linear_delta = ref_lin - calibration_lin
    if linear_ids and abs(proxy_linear_delta) < 1.0e-12:
        raise RuntimeError("calibration-to-reference Linear proxy delta is zero")
    linear_points_per_proxy = (
        online_linear_delta / proxy_linear_delta if linear_ids else 0.0
    )
    attention_points_per_proxy = 0.0
    estimated_score = (
        args.reference_online_score
        + linear_points_per_proxy * (cand_lin - ref_lin)
        + attention_points_per_proxy * (cand_attn - ref_attn)
    )
    estimated_seconds = (
        args.reference_online_seconds
        * candidate["seconds"]
        / reference["seconds"]
    )
    print(
        f"Proxy score: candidate={cand_proxy:.3f} "
        f"(linear={cand_lin:.3f}, attention={cand_attn:.3f}) "
        f"reference={ref_proxy:.3f} "
        f"(linear={ref_lin:.3f}, attention={ref_attn:.3f})"
    )
    print(
        f"Online delta coefficients: linear={linear_points_per_proxy:.6f} "
        f"attention={attention_points_per_proxy:.6f} points/proxy-unit "
        f"linear_anchor={Path(args.linear_calibration_reference).stem}:"
        f"{args.linear_calibration_online_score:.0f}"
    )
    print(
        f"Calibrated online estimate: score={estimated_score:.1f} "
        f"seconds={estimated_seconds:.1f} "
        f"anchor={Path(args.online_reference).stem}:"
        f"{args.reference_online_score:.0f}/"
        f"{args.reference_online_seconds:.0f}s"
    )


if __name__ == "__main__":
    main()
