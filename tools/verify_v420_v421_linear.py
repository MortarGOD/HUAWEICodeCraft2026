"""Paired Linear API checks for v419/v420 equivalence and v421 output descent.

Uses the current evaluator's FP32 baseline GEMM and BF16 player GEMM.
Each version makes its own public-API decisions; this harness never tunes on test data.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import load_solution
from hif4_quant import dequantize_hif4, dequantize_nvfp4, quantize_to_hif4
from self_check import (
    clone_frozen_state,
    validate_hif4_params,
    validate_linear_calibration_result,
)


def assert_tree_equal(expected, actual, path="root", *, allow_extra=False):
    """Exact tensor metadata/values; only dict additions may be allowed."""
    if type(expected) is torch.Tensor:
        assert type(actual) is torch.Tensor, (path, type(actual))
        assert (expected.shape, expected.dtype, expected.device, expected.layout) == (
            actual.shape, actual.dtype, actual.device, actual.layout
        ), (path, "tensor metadata")
        assert torch.equal(expected, actual), (path, "tensor values")
    elif type(expected) is dict:
        assert type(actual) is dict, (path, type(actual))
        assert expected.keys() <= actual.keys(), (path, "missing keys")
        if not allow_extra:
            assert expected.keys() == actual.keys(), (path, "added keys")
        for key, value in expected.items():
            assert_tree_equal(value, actual[key], f"{path}.{key}", allow_extra=allow_extra)
    elif type(expected) in (list, tuple):
        assert type(actual) is type(expected) and len(actual) == len(expected), path
        for i, (left, right) in enumerate(zip(expected, actual)):
            assert_tree_equal(left, right, f"{path}[{i}]", allow_extra=allow_extra)
    else:
        assert type(actual) is type(expected) and actual == expected, path


def tree_extra_keys(expected, actual, path="root"):
    if type(expected) is not dict or type(actual) is not dict:
        return []
    extra = [f"{path}.{key}" for key in actual.keys() - expected.keys()]
    for key in expected.keys() & actual.keys():
        extra.extend(tree_extra_keys(expected[key], actual[key], f"{path}.{key}"))
    return sorted(extra)


def order_for(versions, index):
    orders = list(itertools.permutations(versions))
    return list(orders[index % len(orders)])


def install_counters(module):
    """Counting only: no extra tensor operations inside timed API calls."""
    counts = {"refine_calls": 0, "prefix_calls": 0, "refine_same_object": 0}
    if hasattr(module, "_v421_refine_linear_prefix"):
        original = module._v421_refine_linear_prefix

        def refine(*args, **kwargs):
            counts["refine_calls"] += 1
            result = original(*args, **kwargs)
            counts["refine_same_object"] += int(bool(args) and result is args[0])
            return result

        module._v421_refine_linear_prefix = refine
    if hasattr(module, "_v421_prefix_changes"):
        original_prefix = module._v421_prefix_changes

        def prefix(*args, **kwargs):
            counts["prefix_calls"] += 1
            return original_prefix(*args, **kwargs)

        module._v421_prefix_changes = prefix
    return counts


def counter_delta(after, before):
    return {key: after[key] - before[key] for key in after}


def checked_dynamic(module, packed, state, rng, tag, counts):
    """Cloning/format/immutability checks stay outside the interface timer."""
    frozen = clone_frozen_state(state)
    snapshot = clone_frozen_state(frozen)
    saved_input = clone_frozen_state(packed)
    torch.random.set_rng_state(rng)
    before = dict(counts)
    started = time.perf_counter()
    params = module.hif4_dynamic_quantize_activation(*packed, frozen)
    seconds = time.perf_counter() - started
    assert torch.equal(torch.random.get_rng_state(), rng), (tag, "dynamic RNG changed")
    assert_tree_equal(snapshot, frozen, f"{tag}.state")
    assert_tree_equal(saved_input, packed, f"{tag}.input")
    errors = validate_hif4_params(params, tuple(packed[0].shape), tag)
    assert not errors, (tag, errors)
    return params, seconds, counter_delta(counts, before)


def assert_row_nonincrease(parent, candidate, tag):
    # Errors are recomputed from the same FP32 reference and BF16 GEMM outputs.
    # The tolerance only covers FP32 reduction noise, not BF16 rounding losses.
    tolerance = 8.0 * torch.finfo(torch.float32).eps * parent.abs() + 1.0e-12
    worse = candidate > parent + tolerance
    excess = candidate - parent
    assert not bool(worse.any()), (
        tag, "row MSE regression", torch.nonzero(worse).flatten().tolist(),
        float(excess.max()),
    )
    return {
        "nonincrease": True,
        "rows": int(parent.numel()),
        "improved_rows": int((candidate < parent - tolerance).sum()),
        "equal_within_tolerance_rows": int((excess.abs() <= tolerance).sum()),
        "max_mse_increase": float(excess.max()),
        "max_tolerance": float(tolerance.max()),
    }


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


@torch.inference_mode()
def run(args):
    versions = [int(value) for value in args.versions.split(",")]
    assert versions and len(versions) == len(set(versions)), "Duplicate/empty versions"
    assert set(versions) <= {419, 420, 421}, "Supported versions: 419,420,421"
    assert 421 not in versions or 420 in versions, "v421 requires paired v420"
    torch.set_num_threads(args.threads)
    dataset = (ROOT / args.dataset).resolve()
    output = (ROOT / args.output).resolve()
    source_paths = {v: ROOT / f"solution/solution_v{v}.py" for v in versions}
    hashes = {str(v): hashlib.sha256(path.read_bytes()).hexdigest() for v, path in source_paths.items()}
    bank_path = dataset / "linear.pt"
    bank = torch.load(bank_path, map_location="cpu", weights_only=True)
    selected = list(range(len(bank))) if args.groups == "all" else [int(x) for x in args.groups.split(",")]
    assert selected and len(set(selected)) == len(selected), "Duplicate/empty groups"
    assert all(0 <= index < len(bank) for index in selected), "Invalid group index"
    report = {
        "complete": False, "passed": False, "versions": versions,
        "dataset": str(dataset), "linear_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
        "sha256": hashes, "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch_version": str(torch.__version__), "python": sys.executable,
        "threads": torch.get_num_threads(), "seed": args.seed, "selected_groups": selected,
        "score_prediction_allowed": False,
        "scoring": "FP32 NVFP4 teacher; Algorithm1 FP32 baseline GEMM; decoded BF16 player GEMM then FP32 MSE, as evaluate_local.py",
        "timing": "Serial API wall time; excludes cloning, validation, scoring, fresh-module replay; includes lightweight Python call counters. Not an online-time prediction.",
        "groups": [], "validation_checks": 0,
    }
    write_report(output, report)
    original_rng = torch.random.get_rng_state().clone()
    try:
        for position, gi in enumerate(selected):
            group = bank[gi]
            inputs = (group["weight"], group["calib_activation_list"], group["test_activation_list"])
            saved_inputs = clone_frozen_state(inputs)
            w = dequantize_nvfp4(*group["weight"]).float()
            # Algorithm1 converts its input/parameters to FP32; the shared
            # decoder preserves that dtype. Do not force a BF16 baseline.
            baseline_b = dequantize_hif4(quantize_to_hif4(w), tuple(w.shape))
            assert baseline_b.dtype is torch.float32
            modules = {v: load_solution(source_paths[v]) for v in versions}
            counters = {v: install_counters(modules[v]) for v in versions}
            calibrated, decoded_weights = {}, {}
            item = {
                "group": gi, "shape": list(w.shape),
                "source_model": group.get("source_model"), "source_tensor": group.get("source_tensor"),
                "calibration_order": order_for(versions, position),
                "timing": {str(v): {"calibration": 0.0, "dynamic": 0.0, "isolation": 0.0} for v in versions},
                "calibration_rng_unchanged": {}, "calibration_counts": {},
                "cases": [], "complete": False,
            }
            report["groups"].append(item)
            write_report(output, report)
            torch.manual_seed(args.seed + gi)
            cal_rng = torch.random.get_rng_state().clone()
            cal_end_rng = {}
            for v in item["calibration_order"]:
                torch.random.set_rng_state(cal_rng)
                started = time.perf_counter()
                result = modules[v].hif4_calibration_and_quantize_weight(
                    *group["weight"], group["calib_activation_list"]
                )
                item["timing"][str(v)]["calibration"] = time.perf_counter() - started
                cal_end_rng[v] = torch.random.get_rng_state().clone()
                item["calibration_rng_unchanged"][str(v)] = torch.equal(cal_rng, cal_end_rng[v])
                item["calibration_counts"][str(v)] = dict(counters[v])
                errors = validate_linear_calibration_result(result, tuple(w.shape), f"v{v}.g{gi}.cal")
                assert not errors, (v, gi, errors)
                report["validation_checks"] += 1
                assert_tree_equal(saved_inputs, inputs, f"g{gi}.calibration_inputs")
                calibrated[v] = result
                decoded_weights[v] = dequantize_hif4(result["weight_params"], tuple(w.shape))
                assert decoded_weights[v].dtype is torch.bfloat16
            if 419 in versions and 420 in versions:
                assert_tree_equal(calibrated[419], calibrated[420], f"g{gi}.v419_v420")
                assert torch.equal(cal_end_rng[419], cal_end_rng[420]), "Calibration RNG divergence"
                item["v420_calibration_bitexact"] = True
            if 421 in versions:
                assert_tree_equal(calibrated[420]["weight_params"], calibrated[421]["weight_params"], f"g{gi}.v421_weight")
                assert_tree_equal(calibrated[420]["activation_state"], calibrated[421]["activation_state"], f"g{gi}.v421_parent_state", allow_extra=True)
                assert torch.equal(cal_end_rng[420], cal_end_rng[421]), "Calibration RNG divergence"
                item["v421_parent_state_bitexact"] = True
                item["v421_added_state_fields"] = tree_extra_keys(calibrated[420]["activation_state"], calibrated[421]["activation_state"])
                item["v421_prefix_enabled"] = calibrated[421]["activation_state"].get("v421_prefix_enabled", False)
            saved_calibrated = clone_frozen_state(calibrated)
            first_params = {}
            for ti, packed in enumerate(group["test_activation_list"]):
                a = dequantize_nvfp4(*packed).float()
                target = a @ w.T
                baseline_a = dequantize_hif4(quantize_to_hif4(a), tuple(a.shape))
                assert baseline_a.dtype is torch.float32
                baseline_output = baseline_a @ baseline_b.T
                assert baseline_output.dtype is torch.float32
                baseline_mse = float((target - baseline_output.float()).square().mean())
                assert baseline_mse > 0 and torch.isfinite(torch.tensor(baseline_mse)), (gi, ti, "invalid baseline MSE")
                case = {"test": ti, "rows": int(a.shape[0]), "baseline_mse": baseline_mse,
                        "baseline_output_dtype": str(baseline_output.dtype),
                        "order": order_for(versions, position + ti), "versions": {}}
                item["cases"].append(case)
                params, decoded, row_mse = {}, {}, {}
                torch.manual_seed(args.seed + 10000 + gi * 100 + ti)
                dynamic_rng = torch.random.get_rng_state().clone()
                for v in case["order"]:
                    qa, seconds, calls = checked_dynamic(modules[v], packed, calibrated[v]["activation_state"], dynamic_rng, f"v{v}.g{gi}.t{ti}", counters[v])
                    report["validation_checks"] += 1
                    item["timing"][str(v)]["dynamic"] += seconds
                    params[v] = qa
                    decoded[v] = dequantize_hif4(qa, tuple(a.shape))
                    assert decoded[v].dtype is torch.bfloat16
                    deployed = decoded[v] @ decoded_weights[v].T
                    assert deployed.dtype is torch.bfloat16
                    error = (target - deployed.float()).square()
                    row_mse[v] = error.mean(dim=-1)
                    mse = float(error.mean())
                    case["versions"][str(v)] = {"mse": mse, "score": 1.0 - mse / baseline_mse,
                        "row_mse": row_mse[v].tolist(), "dynamic_seconds": seconds,
                        "output_dtype": str(deployed.dtype), "counts": calls,
                        "state_input_rng_unchanged": True}
                    if ti == 0:
                        first_params[v] = clone_frozen_state(qa)
                if 419 in versions and 420 in versions:
                    assert_tree_equal(params[419], params[420], f"g{gi}.t{ti}.v420_params")
                    assert torch.equal(row_mse[419], row_mse[420]), "v420 output MSE changed"
                    case["v420_bitexact"] = True
                if 421 in versions:
                    case["v421_vs_v420"] = assert_row_nonincrease(row_mse[420], row_mse[421], f"g{gi}.t{ti}")
                    changed = decoded[421] != decoded[420]
                    case["v421_vs_v420"].update(changed_decoded=bool(changed.any()),
                        changed_values=int(changed.sum()), changed_rows=int(changed.any(dim=-1).sum()),
                        score_delta=case["versions"]["421"]["score"] - case["versions"]["420"]["score"])
                assert_tree_equal(saved_calibrated, calibrated, f"g{gi}.calibrated_immutable")
                write_report(output, report)
            # Replay the first input after intervening calls, both in the same
            # module and a fresh module supplied only a cloned frozen state.
            first = group["test_activation_list"][0]
            item["isolation_replays"] = {}
            for v in versions:
                for label, module in (("after_other_inputs", modules[v]), ("fresh_module", load_solution(source_paths[v]))):
                    replay, seconds, _ = checked_dynamic(module, first, calibrated[v]["activation_state"], cal_rng, f"v{v}.g{gi}.{label}", counters[v])
                    report["validation_checks"] += 1
                    assert_tree_equal(first_params[v], replay, f"v{v}.g{gi}.{label}.params")
                    item["timing"][str(v)]["isolation"] += seconds
                item["isolation_replays"][str(v)] = True
            assert_tree_equal(saved_inputs, inputs, f"g{gi}.inputs_immutable")
            assert_tree_equal(saved_calibrated, calibrated, f"g{gi}.final_state_immutable")
            item["score_sums"] = {str(v): sum(c["versions"][str(v)]["score"] for c in item["cases"]) for v in versions}
            item["complete"] = True
            write_report(output, report)
            print(json.dumps({"group": gi, "shape": item["shape"], "scores": item["score_sums"],
                "timing": item["timing"], "v421_changed_cases": sum(c.get("v421_vs_v420", {}).get("changed_decoded", False) for c in item["cases"])}, ensure_ascii=False), flush=True)
        final_hashes = {str(v): hashlib.sha256(path.read_bytes()).hexdigest() for v, path in source_paths.items()}
        assert final_hashes == hashes, "Source changed during paired run"
        report["score_sums"] = {str(v): sum(g["score_sums"][str(v)] for g in report["groups"]) for v in versions}
        report["total_interface_seconds"] = {str(v): sum(g["timing"][str(v)]["calibration"] + g["timing"][str(v)]["dynamic"] for g in report["groups"]) for v in versions}
        report.update(complete=True, passed=True, source_unchanged=True)
    except BaseException as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        write_report(output, report)
        raise
    finally:
        torch.random.set_rng_state(original_rng)
    write_report(output, report)
    print(json.dumps({key: report[key] for key in ("complete", "passed", "score_sums", "total_interface_seconds", "validation_checks")}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="local_fitted_models_attention_feedback_v2")
    parser.add_argument("--groups", default="all", help="all or comma-separated zero-based group IDs")
    parser.add_argument("--output", default="docs/v420-v421-linear-paired.json")
    parser.add_argument("--versions", default="419,420,421", help="Subset of 419,420,421; 421 requires 420")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260907420)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    run(args)


if __name__ == "__main__":
    main()
