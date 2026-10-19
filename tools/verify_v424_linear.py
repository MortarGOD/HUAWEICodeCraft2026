"""Paired public-API Linear evaluation for the v424 learned block transform.

Each module calibrates independently. Up/square regressions remain in the report;
down calibration and dynamic results must be exactly equal. Local score changes
are diagnostics, not online score or runtime predictions.
"""
from __future__ import annotations

import argparse
import ast
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
    install_counters,
    order_for,
    write_report,
)


def install_pair_timer(module):
    """No extra tensor work inside the timed factorization or public API."""
    measurements = {"calls": 0, "seconds": 0.0}
    original = module._pair_polar_factorization

    def measured(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            measurements["seconds"] += time.perf_counter() - started
            measurements["calls"] += 1

    module._pair_polar_factorization = measured
    return measurements


def install_fallback_counter(module, counts):
    counts["polar_fallback_calls"] = 0
    original = getattr(module, "_v422_parent_pair_polar_factorization", None)
    if original is None:
        return

    def counted(*args, **kwargs):
        counts["polar_fallback_calls"] += 1
        return original(*args, **kwargs)

    module._v422_parent_pair_polar_factorization = counted


def install_learning_timers(module, *, required=False):
    """Nested stage timers only; tensor comparison stays outside public APIs."""
    names = (
        "_v424_parent_joint_block_transform",
        "_v424_learn_block_transform",
        "_v424_proxy_pair",
        "_v424_procrustes_step",
        "_v424_two_fold_improves",
    )
    measurements = {}
    for name in names:
        original = getattr(module, name, None)
        if required:
            assert callable(original), (name, "Required candidate instrumentation missing")
        stats = {"present": callable(original), "calls": 0, "seconds": 0.0}
        if name == "_v424_two_fold_improves":
            stats.update(accepted=0, rejected=0)
        if name == "_v424_learn_block_transform":
            stats["returned_parent_object"] = 0
        measurements[name] = stats
        if not callable(original):
            continue

        def measured(*args, _original=original, _stats=stats, _name=name, **kwargs):
            started = time.perf_counter()
            try:
                result = _original(*args, **kwargs)
                if _name == "_v424_two_fold_improves":
                    _stats["accepted" if bool(result) else "rejected"] += 1
                elif _name == "_v424_learn_block_transform" and len(args) >= 3:
                    _stats["returned_parent_object"] += int(result is args[2])
                return result
            finally:
                _stats["seconds"] += time.perf_counter() - started
                _stats["calls"] += 1

        setattr(module, name, measured)
    return measurements


def source_scope(parent_path, candidate_path):
    """Require the exact parent prefix and the joint block transform hook."""
    parent = parent_path.read_bytes()
    candidate = candidate_path.read_bytes()
    assert candidate.startswith(parent), "Candidate does not preserve the complete parent source prefix"
    suffix = ast.parse(candidate[len(parent):].decode("utf-8"))
    bindings = []
    for statement in suffix.body:
        if isinstance(statement, ast.FunctionDef):
            names = [statement.name]
        elif isinstance(statement, ast.Assign):
            assert all(isinstance(target, ast.Name) for target in statement.targets), "Unexpected suffix assignment"
            names = [target.id for target in statement.targets]
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            names = [statement.target.id]
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            continue
        else:
            raise AssertionError((type(statement).__name__, "Unexpected suffix statement"))
        assert all(name.startswith("_v424_") or name == "_v422_joint_block_transform" for name in names), names
        bindings.extend(names)
    assert "_v422_joint_block_transform" in bindings, "Expected block transform hook is absent"
    return {
        "parent_full_prefix_unchanged": True,
        "parent_prefix_bytes": len(parent),
        "suffix_bindings": bindings,
        "attention_functions_redefined": False,
        "interpretation": "Exact parent source plus v424 helpers and the Linear joint block transform hook; pair factorization and public Attention definitions are unchanged. Six-API validation is separate.",
    }


def tree_equal(parent, candidate):
    try:
        assert_tree_equal(parent, candidate)
    except AssertionError:
        return False
    return True


def parameter_changes(parent, candidate):
    assert parent.keys() == candidate.keys(), "HiF4 parameter keys differ"
    fields = {}
    for name in parent:
        left, right = parent[name], candidate[name]
        assert left.shape == right.shape and left.dtype == right.dtype, name
        fields[name] = {"changed_values": int((left != right).sum()), "values": left.numel()}
    return {
        "bitexact": all(field["changed_values"] == 0 for field in fields.values()),
        "fields": fields,
        "interpretation": "Exact changes in the five legal HiF4 parameter fields, not inferred from decoded values.",
    }


def distribution(values):
    values = values.reshape(-1).float()
    quantiles = torch.quantile(values, torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0]))
    return {"count": values.numel(), "mean": float(values.mean()),
            **dict(zip(("min", "p05", "median", "p95", "max"), map(float, quantiles)))}


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


def transform_summary(state):
    transform = state.get("unrestricted_pair_transform")
    if transform is None:
        return {"present": False}
    assert transform.ndim == 3 and transform.shape[-2] == transform.shape[-1]
    width = int(transform.shape[-1])
    identity = torch.eye(width, dtype=transform.dtype, device=transform.device)
    changed = transform != identity
    return {
        "present": True,
        "shape": list(transform.shape),
        "numel": transform.numel(),
        "dtype": str(transform.dtype),
        "block_width": width,
        "blocks": int(transform.shape[0]),
        "nonidentity_blocks": int(changed.flatten(1).any(1).sum()),
        "interpretation": "Nonidentity is structural coverage only, not evidence of learning or quality gain.",
    }


def transform_comparison(parent_state, candidate_state):
    parent = parent_state.get("unrestricted_pair_transform")
    candidate = candidate_state.get("unrestricted_pair_transform")
    if parent is None or candidate is None:
        assert parent is None and candidate is None, "Transform presence differs"
        return {"present": False, "bitexact": True, "changed_blocks": 0}
    assert parent.shape == candidate.shape and parent.dtype == candidate.dtype
    changed = parent != candidate
    delta = candidate.float() - parent.float()
    return {
        "present": True, "bitexact": bool(torch.equal(parent, candidate)),
        "blocks": int(parent.shape[0]),
        "changed_blocks": int(changed.flatten(1).any(1).sum()),
        "changed_values": int(changed.sum()),
        "max_abs_delta": float(delta.abs().max()),
        "frobenius_delta": float(torch.linalg.vector_norm(delta)),
        "interpretation": "Candidate versus actual parent frozen transform; parent nonidentity blocks are not counted as newly learned.",
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
    scope = source_scope(paths[parent], paths[candidate])
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
        "source_scope": scope,
        "torch_version": str(torch.__version__), "python": sys.executable,
        "threads": torch.get_num_threads(), "seed": args.seed, "selected_groups": selected,
        "score_prediction_allowed": False,
        "scoring": "FP32 NVFP4 teacher and Algorithm1 baseline GEMM; actual decoded BF16 player GEMM, then FP32 MSE.",
        "timing": "Serial public API wall time, excluding input/state cloning, checks, scoring and replay. Includes lightweight call counters and nested pair/learning timers. Stage times overlap and must not be summed. No online-time conversion.",
        "validation_policy": "Independent calibrations with identical starting RNG. Down is bitexact; up/square losses may rise and negative cases are retained. Dynamic input/state/RNG must remain unchanged.",
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
            is_down = w.shape[0] < w.shape[1]
            baseline_b = dequantize_hif4(quantize_to_hif4(w.float()), tuple(w.shape))
            assert baseline_b.dtype is torch.float32
            modules = {v: load_solution(paths[v]) for v in versions}
            counters = {v: install_counters(modules[v]) for v in versions}
            for v in versions:
                install_fallback_counter(modules[v], counters[v])
            pair_times = {v: install_pair_timer(modules[v]) for v in versions}
            learning_times = {v: install_learning_timers(modules[v], required=v == candidate) for v in versions}
            calibrated, decoded_weights = {}, {}
            item = {
                "group": gi, "shape": list(w.shape), "is_down": is_down,
                "source_model": group.get("source_model"), "source_tensor": group.get("source_tensor"),
                "calibration_order": order_for(versions, position),
                "timing": {str(v): {"calibration": 0.0, "dynamic": 0.0, "isolation": 0.0} for v in versions},
                "calibration_rng_unchanged": {}, "calibration_counts": {},
                "transforms": {}, "cases": [], "complete": False,
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
                item["timing"][str(v)]["pair_factorization"] = dict(pair_times[v])
                item["timing"][str(v)]["learning_stages"] = {name: dict(stats) for name, stats in learning_times[v].items()}
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
                item["transforms"][str(v)] = transform_summary(result["activation_state"])
            item["calibration_end_rng_equal"] = torch.equal(cal_end_rng[parent], cal_end_rng[candidate])
            item["weight_decoded_changes"] = tensor_changes(decoded_weights[parent], decoded_weights[candidate])
            item["weight_parameter_changes"] = parameter_changes(calibrated[parent]["weight_params"], calibrated[candidate]["weight_params"])
            item["calibration_bitexact"] = tree_equal(calibrated[parent], calibrated[candidate])
            item["transform_comparison"] = transform_comparison(calibrated[parent]["activation_state"], calibrated[candidate]["activation_state"])
            if is_down:
                assert_tree_equal(calibrated[parent], calibrated[candidate], f"g{gi}.down_calibration")
                assert item["calibration_end_rng_equal"], "Down calibration RNG divergence"
                item["down_calibration_bitexact"] = True
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
                case["activation_decoded_changes"] = tensor_changes(decoded[parent], decoded[candidate])
                case["activation_parameter_changes"] = parameter_changes(params[parent], params[candidate])
                case["output_changes"] = tensor_changes(deployed_outputs[parent], deployed_outputs[candidate])
                row_delta = row_mse[candidate] - row_mse[parent]
                tolerance = 8.0 * torch.finfo(torch.float32).eps * row_mse[parent].abs() + 1.0e-12
                case["row_loss_summary"] = {
                    "parent_mse_distribution": distribution(row_mse[parent]),
                    "candidate_mse_distribution": distribution(row_mse[candidate]),
                    "candidate_minus_parent_distribution": distribution(row_delta),
                    "improved": int((row_delta < -tolerance).sum()),
                    "worsened": int((row_delta > tolerance).sum()),
                    "equal_within_tolerance": int((row_delta.abs() <= tolerance).sum()),
                    "max_mse_increase": max(0.0, float(row_delta.max())),
                    "max_mse_decrease": max(0.0, float((-row_delta).max())),
                }
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
            item["score_sums"] = {str(v): sum(c["versions"][str(v)]["score"] for c in item["cases"]) for v in versions}
            item["score_comparison"] = score_comparison(item["score_sums"], parent, candidate)
            item["complete"] = True
            write_report(output, report)
            print(json.dumps({"group": gi, "shape": item["shape"], "score": item["score_comparison"],
                              "timing": item["timing"], "weight_changes": item["weight_decoded_changes"]}, ensure_ascii=False), flush=True)
        final_hashes = {str(v): hashlib.sha256(p.read_bytes()).hexdigest() for v, p in paths.items()}
        assert final_hashes == hashes, "Source changed during paired run"
        final_harness_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        assert final_harness_hash == report["harness_sha256"], "Harness changed during paired run"
        report["final_sha256"] = final_hashes
        report["final_harness_sha256"] = final_harness_hash
        report["score_sums"] = {str(v): sum(g["score_sums"][str(v)] for g in report["groups"]) for v in versions}
        report["score_comparison"] = score_comparison(report["score_sums"], parent, candidate)
        report["total_interface_seconds"] = {
            str(v): sum(g["timing"][str(v)]["calibration"] + g["timing"][str(v)]["dynamic"] for g in report["groups"])
            for v in versions
        }
        report["total_pair_factorization"] = {
            str(v): {field: sum(g["timing"][str(v)]["pair_factorization"][field] for g in report["groups"])
                     for field in ("calls", "seconds")}
            for v in versions
        }
        report["total_learning_stages"] = {}
        for v in versions:
            stages = [g["timing"][str(v)]["learning_stages"] for g in report["groups"]]
            report["total_learning_stages"][str(v)] = {
                name: {
                    field: any(stage[name][field] for stage in stages) if field == "present"
                    else sum(stage[name][field] for stage in stages)
                    for field in stages[0][name]
                }
                for name in stages[0]
            }
        report["transform_coverage"] = {
            "groups_with_changed_transform": sum(not g["transform_comparison"]["bitexact"] for g in report["groups"]),
            "changed_blocks": sum(g["transform_comparison"]["changed_blocks"] for g in report["groups"]),
            "compared_blocks": sum(g["transform_comparison"].get("blocks", 0) for g in report["groups"]),
            "interpretation": "Relative to each independently calibrated parent transform, not relative to identity.",
        }
        deltas = [c["score_comparison"]["delta"] for g in report["groups"] for c in g["cases"]]
        report["case_outcomes"] = {
            "improved": sum(d > 1.0e-12 for d in deltas),
            "worsened": sum(d < -1.0e-12 for d in deltas),
            "equal_within_tolerance": sum(abs(d) <= 1.0e-12 for d in deltas),
            "score_tolerance": 1.0e-12,
        }
        report.update(complete=True, passed=True, source_unchanged=True)
    except BaseException as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        write_report(output, report)
        raise
    finally:
        torch.random.set_rng_state(original_rng)
    write_report(output, report)
    print(json.dumps({key: report[key] for key in (
        "complete", "passed", "score_comparison", "case_outcomes", "total_interface_seconds", "validation_checks"
    )}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=int, default=422)
    parser.add_argument("--candidate", type=int, default=424)
    parser.add_argument("--dataset", default="local_fitted_models_attention_feedback_v2")
    parser.add_argument("--groups", default="all", help="all or comma-separated zero-based group IDs")
    parser.add_argument("--output", default="docs/v424-linear-paired.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260907424)
    args = parser.parse_args()
    if args.threads < 1 or args.parent == args.candidate:
        parser.error("threads must be positive; parent and candidate must differ")
    run(args)


if __name__ == "__main__":
    main()
