#!/usr/bin/env python3
"""Benchmark solutions against the supplied standard HiF4 quantizer.

Unlike ``local_eval.py``, this evaluator does not use another submitted
solution as a proxy baseline.  It implements the scoring flow from
``docs/0819-V2.md`` directly:

    100 * (MSE_standard - MSE_player) / MSE_standard

The standard error for every case is computed once and reused for all
solutions in a run.  Candidate quantization time is measured separately from
dequantization, Linear/Attention execution, and score calculation.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_ids(text: str, count: int) -> list[int]:
    if text.strip().lower() == "all":
        return list(range(count))
    ids = [int(item) for item in text.split(",") if item.strip()]
    invalid = [item for item in ids if item < 0 or item >= count]
    if invalid:
        raise ValueError(f"group ids outside 0..{count - 1}: {invalid}")
    return ids


def mse(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.mean((left.float() - right.float()) ** 2).item())


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    seq = int(q.shape[0])
    q = q.reshape(seq, q_heads, head_dim).transpose(0, 1)
    k = k.reshape(seq, kv_heads, head_dim).transpose(0, 1)
    v = v.reshape(seq, kv_heads, head_dim).transpose(0, 1)
    ratio = q_heads // kv_heads
    if ratio != 1:
        k = k.repeat_interleave(ratio, dim=0)
        v = v.repeat_interleave(ratio, dim=0)
    probs = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(head_dim), dim=-1)
    return (probs @ v).transpose(0, 1).flatten(-2, -1)


def score_case(standard_error: float, player_error: float) -> float:
    if not math.isfinite(standard_error) or standard_error < 0.0:
        raise ValueError(f"invalid standard MSE: {standard_error}")
    # A synthetic extreme case can quantize losslessly under the standard
    # codec.  The published formula is undefined for that case, so keep it
    # out of the comparative score and expose it through ``degenerate_cases``.
    if standard_error == 0.0:
        return 0.0
    return 100.0 * (standard_error - player_error) / standard_error


def timed(call: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    result = call()
    return result, time.perf_counter() - started


@torch.inference_mode()
def linear_standard_errors(data, ids, standard) -> dict[int, list[float]]:
    errors: dict[int, list[float]] = {}
    for position, group_id in enumerate(ids, 1):
        group = data[group_id]
        weight = standard.dequantize_nvfp4(*group["weight"]).float()
        weight_std = standard.dequantize_hif4(
            standard.quantize_to_hif4(weight), tuple(weight.shape)
        ).float()
        group_errors = []
        for packed in group["test_activation_list"]:
            act = standard.dequantize_nvfp4(*packed).float()
            act_std = standard.dequantize_hif4(
                standard.quantize_to_hif4(act), tuple(act.shape)
            ).float()
            group_errors.append(mse(act @ weight.t(), act_std @ weight_std.t()))
        errors[group_id] = group_errors
        print(
            f"baseline linear {position}/{len(ids)} group={group_id}",
            flush=True,
        )
    return errors


@torch.inference_mode()
def attention_standard_errors(data, ids, standard) -> dict[int, list[float]]:
    errors: dict[int, list[float]] = {}
    for position, group_id in enumerate(ids, 1):
        group = data[group_id]
        qh = int(group["q_num_heads"])
        kvh = int(group["kv_num_heads"])
        hd = int(group["head_dim"])
        group_errors = []
        for sample in group["test"]:
            original = {}
            quantized = {}
            for role in ("q", "k", "v"):
                value = standard.dequantize_nvfp4(*sample[role]).float()
                original[role] = value
                quantized[role] = standard.dequantize_hif4(
                    standard.quantize_to_hif4(value), tuple(value.shape)
                ).float()
            reference = attention(
                original["q"], original["k"], original["v"], qh, kvh, hd
            )
            output = attention(
                quantized["q"], quantized["k"], quantized["v"], qh, kvh, hd
            )
            group_errors.append(mse(reference, output))
        errors[group_id] = group_errors
        print(
            f"baseline attention {position}/{len(ids)} group={group_id}",
            flush=True,
        )
    return errors


def new_component() -> dict[str, Any]:
    return {
        "score": 0.0,
        "quant_seconds": 0.0,
        "wall_seconds": 0.0,
        "cases": 0,
        "degenerate_cases": 0,
        "failed_groups": {},
        "groups": {},
    }


@torch.inference_mode()
def evaluate_linear(module, data, ids, standard_errors) -> dict[str, Any]:
    result = new_component()
    wall_started = time.perf_counter()
    for position, group_id in enumerate(ids, 1):
        group = data[group_id]
        packed_weight = group["weight"]
        weight = standard_dequant_nvfp4(*packed_weight).float()
        try:
            calibrated, elapsed = timed(
                lambda: module.hif4_calibration_and_quantize_weight(
                    packed_weight[0],
                    packed_weight[1],
                    group["calib_activation_list"],
                )
            )
        except Exception as error:  # Keep synthetic out-of-domain cases visible.
            result["failed_groups"][str(group_id)] = (
                f"{type(error).__name__}: {error}"
            )
            print(
                f"candidate linear {position}/{len(ids)} group={group_id} "
                f"FAILED {type(error).__name__}: {error}",
                flush=True,
            )
            continue
        result["quant_seconds"] += elapsed
        weight_player = standard_dequant_hif4(
            calibrated["weight_params"], tuple(weight.shape)
        ).float()
        group_scores = []
        group_errors = []
        group_failed = False
        for case_id, packed in enumerate(group["test_activation_list"]):
            act = standard_dequant_nvfp4(*packed).float()
            try:
                params, elapsed = timed(
                    lambda packed=packed: module.hif4_dynamic_quantize_activation(
                        packed[0], packed[1], calibrated["activation_state"]
                    )
                )
            except Exception as error:
                result["failed_groups"][str(group_id)] = (
                    f"case {case_id}: {type(error).__name__}: {error}"
                )
                print(
                    f"candidate linear {position}/{len(ids)} group={group_id} "
                    f"FAILED case={case_id} {type(error).__name__}: {error}",
                    flush=True,
                )
                group_failed = True
                break
            result["quant_seconds"] += elapsed
            act_player = standard_dequant_hif4(params, tuple(act.shape)).float()
            player_error = mse(act @ weight.t(), act_player @ weight_player.t())
            case_score = score_case(
                standard_errors[group_id][case_id], player_error
            )
            result["degenerate_cases"] += int(
                standard_errors[group_id][case_id] == 0.0
            )
            group_scores.append(case_score)
            group_errors.append(player_error)
        if group_failed:
            continue
        result["groups"][str(group_id)] = {
            "score": sum(group_scores),
            "case_scores": group_scores,
            "player_mse": group_errors,
            "standard_mse": standard_errors[group_id],
        }
        result["score"] += sum(group_scores)
        result["cases"] += len(group_scores)
        print(
            f"candidate linear {position}/{len(ids)} group={group_id} "
            f"score={sum(group_scores):.3f}",
            flush=True,
        )
    result["wall_seconds"] = time.perf_counter() - wall_started
    return result


@torch.inference_mode()
def evaluate_attention(module, data, ids, standard_errors) -> dict[str, Any]:
    result = new_component()
    wall_started = time.perf_counter()
    for position, group_id in enumerate(ids, 1):
        group = data[group_id]
        qh = int(group["q_num_heads"])
        kvh = int(group["kv_num_heads"])
        hd = int(group["head_dim"])
        try:
            states, elapsed = timed(
                lambda: module.hif4_calibration_attention(
                    group["calib"], qh, kvh, hd
                )
            )
        except Exception as error:  # Keep synthetic out-of-domain cases visible.
            result["failed_groups"][str(group_id)] = (
                f"{type(error).__name__}: {error}"
            )
            print(
                f"candidate attention {position}/{len(ids)} group={group_id} "
                f"FAILED {type(error).__name__}: {error}",
                flush=True,
            )
            continue
        result["quant_seconds"] += elapsed
        group_scores = []
        group_errors = []
        group_failed = False
        for case_id, sample in enumerate(group["test"]):
            original = {}
            quantized = {}
            for role, function, state, heads in (
                ("q", module.hif4_dynamic_quantize_q, states["q_state"], qh),
                ("k", module.hif4_dynamic_quantize_k, states["k_state"], kvh),
                ("v", module.hif4_dynamic_quantize_v, states["v_state"], kvh),
            ):
                packed = sample[role]
                value = standard_dequant_nvfp4(*packed).float()
                original[role] = value
                try:
                    params, elapsed = timed(
                        lambda packed=packed, function=function, state=state, heads=heads: function(
                            packed[0], packed[1], heads, hd, state
                        )
                    )
                except Exception as error:
                    result["failed_groups"][str(group_id)] = (
                        f"case {case_id} role {role}: "
                        f"{type(error).__name__}: {error}"
                    )
                    print(
                        f"candidate attention {position}/{len(ids)} "
                        f"group={group_id} FAILED case={case_id} role={role} "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
                    group_failed = True
                    break
                result["quant_seconds"] += elapsed
                quantized[role] = standard_dequant_hif4(
                    params, tuple(value.shape)
                ).float()
            if group_failed:
                break
            reference = attention(
                original["q"], original["k"], original["v"], qh, kvh, hd
            )
            output = attention(
                quantized["q"], quantized["k"], quantized["v"], qh, kvh, hd
            )
            player_error = mse(reference, output)
            case_score = score_case(
                standard_errors[group_id][case_id], player_error
            )
            result["degenerate_cases"] += int(
                standard_errors[group_id][case_id] == 0.0
            )
            group_scores.append(case_score)
            group_errors.append(player_error)
        if group_failed:
            continue
        result["groups"][str(group_id)] = {
            "score": sum(group_scores),
            "case_scores": group_scores,
            "player_mse": group_errors,
            "standard_mse": standard_errors[group_id],
        }
        result["score"] += sum(group_scores)
        result["cases"] += len(group_scores)
        print(
            f"candidate attention {position}/{len(ids)} group={group_id} "
            f"score={sum(group_scores):.3f}",
            flush=True,
        )
    result["wall_seconds"] = time.perf_counter() - wall_started
    return result


# Set once in main so evaluation always uses the organizer-supplied codec,
# including when a candidate happens to define helpers with the same names.
standard_dequant_nvfp4: Callable[..., torch.Tensor]
standard_dequant_hif4: Callable[..., torch.Tensor]


def write_checkpoint(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solutions", nargs="+", required=True)
    parser.add_argument("--dataset-dir", default=str(ROOT / "rand_sample"))
    parser.add_argument("--linear-groups", default="all")
    parser.add_argument("--attn-groups", default="all")
    parser.add_argument("--output")
    args = parser.parse_args()

    standard = load_module(ROOT / "hif4_quant.py", "organizer_hif4_quant")
    global standard_dequant_nvfp4, standard_dequant_hif4
    standard_dequant_nvfp4 = standard.dequantize_nvfp4
    standard_dequant_hif4 = standard.dequantize_hif4

    solutions = [Path(item).resolve() for item in args.solutions]
    modules = {
        path.stem: load_module(path, f"candidate_{index}_{path.stem}")
        for index, path in enumerate(solutions)
    }
    if len(modules) != len(solutions):
        raise ValueError("solution basenames must be unique")

    output = Path(args.output).resolve() if args.output else None
    report: dict[str, Any] = {
        "schema": 1,
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "standard_quantizer": str((ROOT / "hif4_quant.py").resolve()),
        "score_scale": 100.0,
        "solutions": {
            name: {"path": str(path)}
            for name, path in zip(modules, solutions)
        },
    }

    dataset_dir = Path(args.dataset_dir)
    linear_data = torch.load(
        dataset_dir / "linear.pt", weights_only=True, map_location="cpu"
    )
    linear_ids = parse_ids(args.linear_groups, len(linear_data))
    report["linear_groups"] = linear_ids
    baseline_started = time.perf_counter()
    linear_baseline = linear_standard_errors(linear_data, linear_ids, standard)
    report["linear_baseline_seconds"] = time.perf_counter() - baseline_started
    linear_valid_ids = [
        group_id
        for group_id in linear_ids
        if all(error > 0.0 for error in linear_baseline[group_id])
    ]
    report["linear_excluded_zero_baseline_groups"] = [
        group_id for group_id in linear_ids if group_id not in linear_valid_ids
    ]
    if len(linear_valid_ids) != len(linear_ids):
        print(
            "excluded zero-baseline linear groups: "
            f"{report['linear_excluded_zero_baseline_groups']}",
            flush=True,
        )
    for name, module in modules.items():
        print(f"=== {name} linear ===", flush=True)
        report["solutions"][name]["linear"] = evaluate_linear(
            module, linear_data, linear_valid_ids, linear_baseline
        )
        write_checkpoint(output, report)
    del linear_data, linear_baseline
    gc.collect()

    attention_data = torch.load(
        dataset_dir / "attn.pt", weights_only=True, map_location="cpu"
    )
    attention_ids = parse_ids(args.attn_groups, len(attention_data))
    report["attention_groups"] = attention_ids
    baseline_started = time.perf_counter()
    attention_baseline = attention_standard_errors(
        attention_data, attention_ids, standard
    )
    report["attention_baseline_seconds"] = time.perf_counter() - baseline_started
    attention_valid_ids = [
        group_id
        for group_id in attention_ids
        if all(error > 0.0 for error in attention_baseline[group_id])
    ]
    report["attention_excluded_zero_baseline_groups"] = [
        group_id for group_id in attention_ids if group_id not in attention_valid_ids
    ]
    if len(attention_valid_ids) != len(attention_ids):
        print(
            "excluded zero-baseline attention groups: "
            f"{report['attention_excluded_zero_baseline_groups']}",
            flush=True,
        )
    for name, module in modules.items():
        print(f"=== {name} attention ===", flush=True)
        component = evaluate_attention(
            module, attention_data, attention_valid_ids, attention_baseline
        )
        solution = report["solutions"][name]
        solution["attention"] = component
        solution["score"] = solution["linear"]["score"] + component["score"]
        solution["quant_seconds"] = (
            solution["linear"]["quant_seconds"] + component["quant_seconds"]
        )
        solution["wall_seconds"] = (
            solution["linear"]["wall_seconds"] + component["wall_seconds"]
        )
        print(
            f"RESULT {name} score={solution['score']:.3f} "
            f"linear={solution['linear']['score']:.3f} "
            f"attention={component['score']:.3f} "
            f"quant_seconds={solution['quant_seconds']:.3f} "
            f"wall_seconds={solution['wall_seconds']:.3f}",
            flush=True,
        )
        write_checkpoint(output, report)

    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
