"""Paired public-API Linear evaluation for the v423 correlated-channel pair search candidate.

Each module calibrates independently and the complete calibration must be equal.
Actual decoded BF16 GEMM row regressions are retained, and make passed=false;
down dynamic results must be exactly equal. Local scores are diagnostics only.
Run dataset estimates only after the candidate has passed self_check and is pushed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import load_solution
from hif4_quant import dequantize_hif4, dequantize_nvfp4, quantize_to_hif4
from self_check import clone_frozen_state, validate_linear_calibration_result
from tools.verify_v420_v421_linear import (
    assert_tree_equal,
    checked_dynamic,
    install_counters as install_parent_counters,
    order_for,
    write_report,
)


def install_search_counters(module):
    """Counting and wall time only; no tensor work added inside timed APIs."""
    counts = install_parent_counters(module)
    symbols = ("_v423_refine_linear_pairs", "_v423_pair_code_proposals")
    installed = []
    for symbol in symbols:
        original = getattr(module, symbol, None)
        if not callable(original):
            continue
        installed.append(symbol)
        prefix = symbol.removeprefix("_")
        counts[prefix + "_calls"] = 0
        counts[prefix + "_seconds"] = 0.0
        counts[prefix + "_same_object"] = 0

        def counted(*args, _original=original, _prefix=prefix, **kwargs):
            started = time.perf_counter()
            counts[_prefix + "_calls"] += 1
            try:
                result = _original(*args, **kwargs)
                counts[_prefix + "_same_object"] += int(bool(args) and result is args[0])
                return result
            finally:
                counts[_prefix + "_seconds"] += time.perf_counter() - started

        setattr(module, symbol, counted)
    return counts, installed


def parameter_changes(parent, candidate, logical_shape):
    """Representational code changes and exact signed-code changes separately."""
    rows = 1
    for dim in logical_shape[:-1]:
        rows *= int(dim)
    changed = ((parent["sign"] != candidate["sign"])
               | (parent["mant"] != candidate["mant"])).reshape(rows, -1)
    signed_changed = (parent["sign"] * parent["mant"]
                      != candidate["sign"] * candidate["mant"]).reshape(rows, -1)
    return {
        "changed_codes": int(changed.sum()), "codes": changed.numel(),
        "changed_rows": int(changed.any(-1).sum()),
        "changed_codes_per_row": changed.sum(-1).tolist(),
        "changed_signed_codes": int(signed_changed.sum()),
        "fields": {key: {"changed_values": int((parent[key] != candidate[key]).sum()),
                         "values": parent[key].numel()}
                   for key in parent},
    }


def tensor_changes(parent, candidate):
    """Exact decoded values, with no inference that changed means improved."""
    assert parent.shape == candidate.shape and parent.dtype == candidate.dtype
    changed = parent != candidate
    return {
        "changed_values": int(changed.sum()),
        "values": parent.numel(),
        "changed_rows": int(changed.reshape(-1, changed.shape[-1]).any(-1).sum()),
        "any_changed": bool(changed.any()),
    }


def score_comparison(scores, parent, candidate):
    return {
        "parent": scores[str(parent)],
        "candidate": scores[str(candidate)],
        "delta": scores[str(candidate)] - scores[str(parent)],
    }


@torch.inference_mode()
def run(args):
    parent, candidate = args.parent, args.candidate
    assert parent != candidate, "Parent and candidate must be different versions"
    versions = [parent, candidate]
    torch.set_num_threads(args.threads)
    dataset = (ROOT / args.dataset).resolve()
    output = (ROOT / args.output).resolve()
    paths = {v: ROOT / f"solution/solution_v{v}.py" for v in versions}
    hashes = {str(v): hashlib.sha256(p.read_bytes()).hexdigest() for v, p in paths.items()}
    bank_path = dataset / "linear.pt"
    bank = torch.load(bank_path, map_location="cpu", weights_only=True)
    selected = list(range(len(bank))) if args.groups == "all" else [int(x) for x in args.groups.split(",")]
    assert selected and len(selected) == len(set(selected)), "Duplicate/empty groups"
    assert all(0 <= gi < len(bank) for gi in selected), "Invalid group index"
    report = {
        "complete": False, "passed": False, "parent": parent, "candidate": candidate,
        "versions": versions, "dataset": str(dataset), "sha256": hashes,
        "linear_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch_version": str(torch.__version__), "python": sys.executable,
        "threads": torch.get_num_threads(), "seed": args.seed, "selected_groups": selected,
        "score_prediction_allowed": False,
        "candidate_search": "Correlated channel pairs; 15 signed codes per channel yield 225 pair candidates; actual decoded BF16 output is the final acceptance target.",
        "scoring": "FP32 NVFP4 teacher and Algorithm1 baseline GEMM; actual decoded BF16 player GEMM, then FP32 MSE.",
        "timing": "Serial public API wall time, excluding input/state cloning, checks, scoring and replay. Includes lightweight call counters and v423 search timers; search time is a subset of dynamic time. No online-time conversion.",
        "validation_policy": "Independent calibrations with identical starting RNG must be fully bitexact. Down is bitexact. Every actual BF16 output row is checked for exact nonincrease; any positive row MSE delta is retained and makes passed false. Tolerance is diagnostic only. Dynamic input/state/RNG must remain unchanged.",
        "groups": [], "validation_checks": 0, "row_nonincrease_failures": [],
        "graph_evidence": "Source-read fallback for this new harness; no graph verification claimed for unindexed new code.",
    }
    write_report(output, report)
    original_rng = torch.random.get_rng_state().clone()
    try:
        for position, gi in enumerate(selected):
            group = bank[gi]
            inputs = (group["weight"], group["calib_activation_list"], group["test_activation_list"])
            saved_inputs = clone_frozen_state(inputs)
            w = dequantize_nvfp4(*group["weight"]).float()
            is_down = w.shape[0] < w.shape[1]
            baseline_b = dequantize_hif4(quantize_to_hif4(w.float()), tuple(w.shape))
            assert baseline_b.dtype is torch.float32
            modules = {v: load_solution(paths[v]) for v in versions}
            instrumentation = {v: install_search_counters(modules[v]) for v in versions}
            counters = {v: instrumentation[v][0] for v in versions}
            calibrated, decoded_weights = {}, {}
            item = {
                "group": gi, "shape": list(w.shape), "is_down": is_down,
                "source_model": group.get("source_model"), "source_tensor": group.get("source_tensor"),
                "calibration_order": order_for(versions, position),
                "timing": {str(v): {"calibration": 0.0, "dynamic": 0.0, "isolation": 0.0} for v in versions},
                "calibration_rng_unchanged": {}, "calibration_counts": {},
                "search_symbols": {str(v): instrumentation[v][1] for v in versions},
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
            item["calibration_end_rng_equal"] = torch.equal(cal_end_rng[parent], cal_end_rng[candidate])
            item["weight_decoded_changes"] = tensor_changes(decoded_weights[parent], decoded_weights[candidate])
            assert_tree_equal(calibrated[parent], calibrated[candidate], f"g{gi}.complete_calibration")
            assert item["calibration_end_rng_equal"], "Calibration RNG divergence"
            item["calibration_bitexact"] = True
            report["validation_checks"] += 1
            saved_calibrated = clone_frozen_state(calibrated)
            first_params = {}
            assert group["test_activation_list"], (gi, "No test inputs")
            for ti, packed in enumerate(group["test_activation_list"]):
                a = dequantize_nvfp4(*packed).float()
                target = a @ w.T
                baseline_a = dequantize_hif4(quantize_to_hif4(a.float()), tuple(a.shape))
                assert baseline_a.dtype is torch.float32
                baseline_output = baseline_a @ baseline_b.T
                assert baseline_output.dtype is torch.float32
                baseline_mse = float((target - baseline_output).square().mean())
                assert baseline_mse > 0 and torch.isfinite(torch.tensor(baseline_mse)), (gi, ti, "Invalid baseline MSE")
                case = {
                    "test": ti, "rows": a.numel() // a.shape[-1], "baseline_mse": baseline_mse,
                    "baseline_output_dtype": str(baseline_output.dtype),
                    "order": order_for(versions, position + ti), "versions": {},
                }
                item["cases"].append(case)
                params, decoded, row_mse, deployed_outputs = {}, {}, {}, {}
                torch.manual_seed(args.seed + 10000 + gi * 100 + ti)
                dynamic_rng = torch.random.get_rng_state().clone()
                for v in case["order"]:
                    qa, seconds, calls = checked_dynamic(
                        modules[v], packed, calibrated[v]["activation_state"], dynamic_rng,
                        f"v{v}.g{gi}.t{ti}", counters[v],
                    )
                    report["validation_checks"] += 1
                    item["timing"][str(v)]["dynamic"] += seconds
                    params[v] = qa
                    decoded[v] = dequantize_hif4(qa, tuple(a.shape))
                    assert decoded[v].dtype is torch.bfloat16
                    deployed = decoded[v] @ decoded_weights[v].T
                    assert deployed.dtype is torch.bfloat16
                    deployed_outputs[v] = deployed
                    error = (target - deployed.float()).square()
                    row_mse[v] = error.reshape(-1, error.shape[-1]).mean(-1)
                    mse = float(error.mean())
                    assert torch.isfinite(torch.tensor(mse)), (gi, ti, v, "Nonfinite player MSE")
                    case["versions"][str(v)] = {
                        "mse": mse, "score": 1.0 - mse / baseline_mse,
                        "dynamic_seconds": seconds, "output_dtype": str(deployed.dtype),
                        "counts": calls, "state_input_rng_unchanged": True,
                    }
                    if ti == 0:
                        first_params[v] = clone_frozen_state(qa)
                scores = {str(v): case["versions"][str(v)]["score"] for v in versions}
                case["score_comparison"] = score_comparison(scores, parent, candidate)
                case["activation_parameter_changes"] = parameter_changes(params[parent], params[candidate], a.shape)
                case["activation_decoded_changes"] = tensor_changes(decoded[parent], decoded[candidate])
                case["output_changes"] = tensor_changes(deployed_outputs[parent], deployed_outputs[candidate])
                row_delta = row_mse[candidate] - row_mse[parent]
                tolerance = 8.0 * torch.finfo(torch.float32).eps * row_mse[parent].abs() + 1.0e-12
                case["row_loss_summary"] = {
                    "improved": int((row_delta < -tolerance).sum()),
                    "worsened": int((row_delta > tolerance).sum()),
                    "equal_within_tolerance": int((row_delta.abs() <= tolerance).sum()),
                    "max_mse_increase": max(0.0, float(row_delta.max())),
                    "max_mse_decrease": max(0.0, float((-row_delta).max())),
                    "max_tolerance": float(tolerance.max()),
                    "nonincrease": not bool((row_delta > 0).any()),
                    "strict_worsened": int((row_delta > 0).sum()),
                }
                worse_indices = torch.nonzero(row_delta > 0, as_tuple=False).flatten().tolist()
                case["negative_rows"] = [
                    {"row": ri, "parent_mse": float(row_mse[parent][ri]),
                     "candidate_mse": float(row_mse[candidate][ri]),
                     "mse_increase": float(row_delta[ri]), "tolerance": float(tolerance[ri])}
                    for ri in worse_indices
                ]
                if worse_indices:
                    report["row_nonincrease_failures"].append(
                        {"group": gi, "test": ti, "rows": case["negative_rows"]})
                report["validation_checks"] += 1
                if is_down:
                    assert_tree_equal(params[parent], params[candidate], f"g{gi}.t{ti}.down_params")
                    assert torch.equal(deployed_outputs[parent], deployed_outputs[candidate]), "Down output changed"
                    case["down_dynamic_bitexact"] = True
                assert_tree_equal(saved_calibrated, calibrated, f"g{gi}.calibrated_immutable")
                write_report(output, report)
            item["isolation_replays"] = {}
            first = group["test_activation_list"][0]
            for v in versions:
                for label, module in (("after_other_inputs", modules[v]), ("fresh_module", load_solution(paths[v]))):
                    replay, seconds, _ = checked_dynamic(
                        module, first, calibrated[v]["activation_state"], cal_rng,
                        f"v{v}.g{gi}.{label}", counters[v],
                    )
                    report["validation_checks"] += 1
                    assert_tree_equal(first_params[v], replay, f"v{v}.g{gi}.{label}.params")
                    item["timing"][str(v)]["isolation"] += seconds
                item["isolation_replays"][str(v)] = True
            assert_tree_equal(saved_inputs, inputs, f"g{gi}.inputs_immutable")
            assert_tree_equal(saved_calibrated, calibrated, f"g{gi}.final_state_immutable")
            for v in versions:
                item["timing"][str(v)]["v423_search_seconds"] = sum(
                    c["versions"][str(v)]["counts"].get("v423_refine_linear_pairs_seconds", 0.0)
                    for c in item["cases"]
                )
            item["dynamic_counts"] = {
                str(v): {key: sum(c["versions"][str(v)]["counts"][key] for c in item["cases"])
                         for key in item["cases"][0]["versions"][str(v)]["counts"]}
                for v in versions
            }
            item["score_sums"] = {str(v): sum(c["versions"][str(v)]["score"] for c in item["cases"]) for v in versions}
            item["score_comparison"] = score_comparison(item["score_sums"], parent, candidate)
            item["complete"] = True
            write_report(output, report)
            print(json.dumps({"group": gi, "shape": item["shape"], "score": item["score_comparison"],
                              "timing": item["timing"], "weight_changes": item["weight_decoded_changes"]}, ensure_ascii=False), flush=True)
        final_hashes = {str(v): hashlib.sha256(p.read_bytes()).hexdigest() for v, p in paths.items()}
        report["sha256_after"] = final_hashes
        assert final_hashes == hashes, "Source changed during paired run"
        report["score_sums"] = {str(v): sum(g["score_sums"][str(v)] for g in report["groups"]) for v in versions}
        report["score_comparison"] = score_comparison(report["score_sums"], parent, candidate)
        report["total_interface_seconds"] = {
            str(v): sum(g["timing"][str(v)]["calibration"] + g["timing"][str(v)]["dynamic"] for g in report["groups"])
            for v in versions
        }
        deltas = [c["score_comparison"]["delta"] for g in report["groups"] for c in g["cases"]]
        report["case_outcomes"] = {
            "improved": sum(d > 1.0e-12 for d in deltas),
            "worsened": sum(d < -1.0e-12 for d in deltas),
            "equal_within_tolerance": sum(abs(d) <= 1.0e-12 for d in deltas),
            "score_tolerance": 1.0e-12,
        }
        report["row_outcomes"] = {
            key: sum(c["row_loss_summary"][key] for g in report["groups"] for c in g["cases"])
            for key in ("improved", "worsened", "equal_within_tolerance", "strict_worsened")
        }
        report.update(complete=True, passed=report["row_outcomes"]["strict_worsened"] == 0,
                      contract_checks_passed=True, source_unchanged=True)
    except BaseException as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        write_report(output, report)
        raise
    finally:
        torch.random.set_rng_state(original_rng)
    write_report(output, report)
    print(json.dumps({key: report[key] for key in (
        "complete", "passed", "score_comparison", "case_outcomes", "row_outcomes", "total_interface_seconds", "validation_checks"
    )}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=int, default=422)
    parser.add_argument("--candidate", type=int, default=423)
    parser.add_argument("--dataset", default="local_fitted_models_attention_feedback_v2")
    parser.add_argument("--groups", default="all", help="all or comma-separated zero-based group IDs")
    parser.add_argument("--output", default="docs/v423-linear-paired.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260907423)
    args = parser.parse_args()
    if args.threads < 1 or args.parent == args.candidate:
        parser.error("threads must be positive; parent and candidate must differ")
    report = run(args)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
