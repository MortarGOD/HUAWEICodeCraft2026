"""Paired public-API Linear evaluation for the L2/L3 complete deployment candidates.

Each module calibrates independently. Up/square regressions remain in the report;
down calibration and dynamic results must be exactly equal. Local score changes
are diagnostics, not online score or runtime predictions.
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


def install_campaign(module):
    records={}
    for name in ('_lr_joint_probe','_lr_rotation_probe','_lr_joint_proposal','_lr_rotation_gradient','_lr_rotation_trial','_lr_compile','_lr_evaluate','_lr_final_guard'):
        fn=getattr(module,name,None)
        if fn is None: continue
        rec={'calls':0,'seconds':0.,'exceptions':[],'losses':[],'accepted':[]}
        records[name]=rec
        def wrapped(*a,__fn=fn,__name=name,__rec=rec,**kw):
            t=time.perf_counter()
            try:
                out=__fn(*a,**kw)
            except Exception as exc:
                __rec['exceptions'].append(type(exc).__name__+': '+str(exc))
                raise
            finally:
                __rec['calls']+=1;__rec['seconds']+=time.perf_counter()-t
            if __name=='_lr_evaluate': __rec['losses'].append(out[0].tolist())
            if __name=='_lr_final_guard': __rec['accepted'].append(out)
            if __name.endswith('_probe'): __rec['accepted'].append(out is not a[0])
            return out
        setattr(module,name,wrapped)
    return records


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
        "scoring": "FP32 NVFP4 teacher and Algorithm1 baseline GEMM; actual decoded BF16 player GEMM, then FP32 MSE.",
        "timing": "Serial public API wall time, excluding input/state cloning, checks, scoring and replay. Includes lightweight call counters and pair-factorization timer; pair time is a subset of calibration time. No online-time conversion.",
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
            campaigns = {v: install_campaign(modules[v]) for v in versions}
            for v in versions:
                install_fallback_counter(modules[v], counters[v])
            pair_times = {v: install_pair_timer(modules[v]) for v in versions}
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
                cal_end_rng[v] = torch.random.get_rng_state().clone()
                item["calibration_rng_unchanged"][str(v)] = torch.equal(cal_rng, cal_end_rng[v])
                item["calibration_counts"][str(v)] = dict(counters[v])
                item.setdefault("campaign",{})[str(v)]=campaigns[v]
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
                case["output_changes"] = tensor_changes(deployed_outputs[parent], deployed_outputs[candidate])
                row_delta = row_mse[candidate] - row_mse[parent]
                tolerance = 8.0 * torch.finfo(torch.float32).eps * row_mse[parent].abs() + 1.0e-12
                case["row_loss_summary"] = {
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
    parser.add_argument("--parent", type=int, default=421)
    parser.add_argument("--candidate", type=int, default=422)
    parser.add_argument("--dataset", default="local_fitted_models_attention_feedback_v2")
    parser.add_argument("--groups", default="all", help="all or comma-separated zero-based group IDs")
    parser.add_argument("--output", default="docs/v422-linear-paired.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260907422)
    args = parser.parse_args()
    if args.threads < 1 or args.parent == args.candidate:
        parser.error("threads must be positive; parent and candidate must differ")
    run(args)


if __name__ == "__main__":
    main()
