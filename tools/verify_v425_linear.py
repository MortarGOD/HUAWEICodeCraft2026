"""Paired public-API Linear evaluation for the v425 sparse weight microgroup calibration.

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


def install_calibration_instrumentation(module, *, required=False):
    """Record timers and references only; all tensor audits run after calibration.

    Do not retain microgroup gradients/Hessians: they are intentionally chunked.
    Retained proposals are O(output rows), and fixed-Z loss arrays are 2 x O.
    """
    names = (
        "_v425_parent_calibration", "_v425_microgroup_proposals",
        "_v425_fixed_z_losses", "_v425_fixed_z_refine",
        "_v425_recompile_state", "_v425_last_calibration_guard",
        "_v425_calibration_probe",
    )
    measurements, stack = {}, []
    capture = {"proposals": [], "fixed_z_losses": [], "dynamic_calls": [],
               "exceptions": [], "refinements": [], "recompilations": [],
               "guards": [], "probes": []}
    for name in names:
        original = getattr(module, name, None)
        if required:
            assert callable(original), (name, "Required candidate instrumentation missing")
        stats = {"present": callable(original), "calls": 0, "seconds": 0.0, "exceptions": 0}
        measurements[name] = stats
        if not callable(original):
            continue

        def measured(*args, _original=original, _stats=stats, _name=name, **kwargs):
            stack.append(_name)
            started = time.perf_counter()
            try:
                result = _original(*args, **kwargs)
                if _name == "_v425_microgroup_proposals":
                    capture["proposals"].append(result)
                elif _name == "_v425_fixed_z_losses":
                    capture["fixed_z_losses"].append(result)
                elif _name == "_v425_fixed_z_refine":
                    capture["refinements"].append({"parent_params": args[0], "samples": args[1], "result": result})
                elif _name == "_v425_recompile_state":
                    capture["recompilations"].append({"parent_state": args[0], "params": args[1], "result": result})
                elif _name == "_v425_last_calibration_guard":
                    capture["guards"].append(result)
                elif _name == "_v425_calibration_probe":
                    capture["probes"].append({"parent": args[0], "result": result})
                return result
            except BaseException as exc:
                _stats["exceptions"] += 1
                capture["exceptions"].append({"helper": _name, "type": type(exc).__name__})
                if _name == "_v425_calibration_probe":
                    capture["probes"].append({"parent": args[0], "result": None})
                raise
            finally:
                _stats["seconds"] += time.perf_counter() - started
                _stats["calls"] += 1
                stack.pop()

        setattr(module, name, measured)
    if required:
        original_dynamic = module.hif4_dynamic_quantize_activation

        def dynamic(*args, **kwargs):
            if "_v425_calibration_probe" not in stack:
                return original_dynamic(*args, **kwargs)
            stage = "encoder_guard" if "_v425_last_calibration_guard" in stack else "sampled_parent_fit"
            started = time.perf_counter()
            try:
                return original_dynamic(*args, **kwargs)
            finally:
                capture["dynamic_calls"].append({"stage": stage, "shape": list(args[0].shape),
                                                 "seconds": time.perf_counter() - started})

        module.hif4_dynamic_quantize_activation = dynamic
    return measurements, capture


def source_scope(parent_path, candidate_path):
    """Require the exact parent prefix and only a new public Linear calibration wrapper."""
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
        assert all(name.startswith("_v425_") or name == "hif4_calibration_and_quantize_weight" for name in names), names
        bindings.extend(names)
    assert "hif4_calibration_and_quantize_weight" in bindings, "Expected Linear calibration wrapper is absent"
    return {
        "parent_full_prefix_unchanged": True,
        "parent_prefix_bytes": len(parent),
        "suffix_bindings": bindings,
        "attention_functions_redefined": False,
        "linear_dynamic_function_redefined": False,
        "interpretation": "Exact parent source plus private v425 helpers and the public Linear calibration wrapper. Linear dynamic and all Attention definitions are unchanged; numerical Attention validation is separate.",
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


def sparse_parameter_changes(parent, candidate):
    """Check the actual legal metadata/code move, independently of decoded W."""
    changes = parameter_changes(parent, candidate)
    for field in ("scale_factor", "scale_lv2"):
        assert_tree_equal(parent[field], candidate[field], f"fixed_weight.{field}")
    rows = int(parent["mant"].shape[0])
    old_lv3 = parent["scale_lv3"].reshape(rows, -1)
    new_lv3 = candidate["scale_lv3"].reshape_as(old_lv3)
    lv3_changed = old_lv3 != new_lv3
    groups_per_row = lv3_changed.sum(-1)
    assert bool((groups_per_row <= 1).all()), "More than one changed lv3 microgroup per output row"
    assert bool(((old_lv3[lv3_changed] == 1) & (new_lv3[lv3_changed] == 2)
                 | (old_lv3[lv3_changed] == 2) & (new_lv3[lv3_changed] == 1)).all()), "lv3 move is not a 1/2 flip"
    metadata_code_changes = torch.zeros_like(lv3_changed)
    for field in ("sign", "mant"):
        changed = (parent[field] != candidate[field]).reshape(rows, -1, 4).any(-1)
        metadata_code_changes |= changed
        assert not bool((changed & ~lv3_changed).any()), (field, "Code changed outside flipped lv3 group")
    old_codes = (parent["sign"].float() * parent["mant"].float() * 4).reshape(rows, -1, 4)
    new_codes = (candidate["sign"].float() * candidate["mant"].float() * 4).reshape_as(old_codes)
    assert bool((new_codes == new_codes.round()).all() and (new_codes.abs() <= 7).all()), "Illegal signed W codes"
    changes.update(
        fixed_scale_factor=True, fixed_scale_lv2=True,
        changed_lv3_groups=int(lv3_changed.sum()),
        changed_lv3_rows=int((groups_per_row > 0).sum()),
        max_changed_lv3_groups_per_row=int(groups_per_row.max()),
        changed_code_values=int((old_codes != new_codes).sum()),
        changed_code_groups=int((old_codes != new_codes).any(-1).sum()),
        changed_sign_or_mant_groups=int(metadata_code_changes.sum()),
        changed_output_rows=torch.nonzero(groups_per_row, as_tuple=False).flatten().tolist(),
    )
    return changes


def compiled_state_changes(parent_state, candidate_state, params, module):
    """All coordinates/teachers stay exact; only decoder-derived fields may vary."""
    allowed = {
        "unrestricted_deployed_weight", "unrestricted_continuous_projection",
        "unrestricted_block_gram", "unrestricted_block_inverse",
        "unrestricted_micro_inverse", "linear_full_gram",
        "unrestricted_super_gram", "unrestricted_super_inverse",
    }
    all_keys = parent_state.keys() | candidate_state.keys()
    changed = []
    for name in sorted(all_keys):
        if name in parent_state and name in candidate_state and tree_equal(parent_state[name], candidate_state[name]):
            continue
        assert name in allowed, (name, "Coordinate, teacher, or unrelated state changed")
        changed.append(name)
    decoder_matches = None
    if "unrestricted_deployed_weight" in candidate_state:
        decoder_matches = torch.equal(candidate_state["unrestricted_deployed_weight"], module._linear_score_decode(params))
        if changed:
            assert decoder_matches, "Recompiled decoder differs from final legal BF16 W"
    return {"changed_fields": changed, "allowed_changed_fields": sorted(allowed),
            "coordinate_teacher_other_state_bitexact": True,
            "compiled_decoder_matches_legal_weight": decoder_matches,
            "interpretation": "Exact comparison of every state key. Decoder/Gram/projection fields may change; the Gram and solve arithmetic itself is covered by separate mechanism checks."}


def loss_summary(before, after):
    assert before.shape == after.shape and before.shape[0] == 2
    left, right = before.reshape(2, -1).float(), after.reshape(2, -1).float()
    left_sum, right_sum = left.sum(-1), right.sum(-1)
    def finite_list(values):
        return [float(value) if bool(torch.isfinite(value)) else None for value in values]
    return {
        "before_fold_sums": finite_list(left_sum), "after_fold_sums": finite_list(right_sum),
        "fold_ratios": finite_list(right_sum / left_sum.clamp_min(1.0e-30)),
        "nonfinite_before_values": int((~torch.isfinite(left)).sum()),
        "nonfinite_after_values": int((~torch.isfinite(right)).sum()),
        "compared_columns": int(left.shape[1]),
        "both_folds_strictly_improved_columns": int((right < left).all(0).sum()),
        "any_fold_worsened_columns": int((right > left).any(0).sum()),
        "bitexact_columns": int((right == left).all(0).sum()),
        "max_loss_increase": max(0.0, float((right-left).max())) if bool(torch.isfinite(right-left).all()) else None,
    }


def audit_sparse_calibration(module, capture, independent_parent, final_result):
    """Tensor inspection is deliberately outside all reported public API timers."""
    assert len(capture["probes"]) == 1, "Expected one candidate probe after complete parent calibration"
    probe = capture["probes"][0]
    parent = probe["parent"]
    assert_tree_equal(independent_parent, parent, "captured_parent_vs_independent_parent")
    parent_params = parent["weight_params"]
    outputs = int(parent_params["mant"].shape[0])
    parent_fallback = final_result is parent
    if parent_fallback:
        assert_tree_equal(independent_parent, final_result, "final_parent_fallback")
    else:
        assert probe["result"] is final_result, "Unexpected result bypassed the probe"

    proposed = {name: value.clone() for name, value in parent_params.items()}
    mant = proposed["mant"].reshape(outputs, -1, 4)
    sign = proposed["sign"].reshape_as(mant)
    lv3 = proposed["scale_lv3"].reshape(outputs, -1)
    row_start, shortlisted, selected = 0, 0, 0
    predicted_improvement = torch.zeros(2, dtype=torch.float32)
    for proposal in capture["proposals"]:
        take = proposal["take"]
        count = int(take.numel())
        assert count <= 128 and row_start + count <= outputs
        rows = torch.nonzero(take, as_tuple=False).flatten()
        groups, codes = proposal["groups"][rows], proposal["codes"][rows]
        mant[row_start+rows, groups] = codes.abs().to(mant.dtype) / 4
        sign[row_start+rows, groups] = codes.sign().to(sign.dtype)
        lv3[row_start+rows, groups] = proposal["lv3"][rows].to(lv3.dtype)
        assert bool((proposal["shortlist_count"] <= 4).all()), "Shortlist exceeds four groups per row"
        shortlisted += int(proposal["shortlist_count"].sum())
        selected += int(rows.numel())
        predicted_improvement += proposal["improvement"][:, take].float().sum(-1)
        row_start += count
    if capture["refinements"]:
        assert row_start == outputs, "Proposal chunks did not cover all output rows"
    report = {
        "captured_parent_matches_independent_parent": True,
        "parent_fallback": parent_fallback, "final_accepted": not parent_fallback,
        "exceptions": capture["exceptions"],
        "proposals": {"chunks": len(capture["proposals"]), "output_rows_covered": row_start,
                      "shortlisted_groups": shortlisted, "selected_rows": selected,
                      "predicted_fp32_improvement_per_fold": predicted_improvement.tolist(),
                      "parameters": sparse_parameter_changes(parent_params, proposed)},
        "fixed_z": {"called": bool(capture["refinements"]), "accepted_rows": 0},
        "recompile": {"called": bool(capture["recompilations"]), "succeeded": False},
        "encoder_guard": {"called": bool(capture["guards"]), "accepted": False},
        "final_parameters": sparse_parameter_changes(parent_params, final_result["weight_params"]),
        "final_state": compiled_state_changes(parent["activation_state"], final_result["activation_state"], final_result["weight_params"], module),
        "additional_dynamic_calls": list(capture["dynamic_calls"]),
        "limitations": "FP32 fixed-Z proposal gains include within-group cross terms. BF16 fixed-Z and recompiled-encoder guards are separate. Sampled input shapes can choose a different dynamic budget; the last calibration sequence was already available to the parent and is not a global holdout. All tensor audits and proposal reconstruction occur after the interface timer.",
    }
    sampled_calls = [entry for entry in capture["dynamic_calls"] if entry["stage"] == "sampled_parent_fit"]
    guard_calls = [entry for entry in capture["dynamic_calls"] if entry["stage"] == "encoder_guard"]
    assert len(sampled_calls) <= 4 and all(entry["shape"][0] <= 32 for entry in sampled_calls)
    assert len(guard_calls) <= 2
    reason = "ineligible_or_early_parent_fallback"
    if capture["refinements"]:
        assert len(capture["refinements"]) == 1
        refinement = capture["refinements"][0]
        result = refinement["result"]
        report["fixed_z"]["sample_shapes"] = [list(z.shape) for z, _, _ in refinement["samples"]]
        report["fixed_z"]["normalizers"] = [float(n) for _, _, n in refinement["samples"]]
        report["fixed_z"]["loss_units"] = "Per-sequence SSE / target energy, summed in calibration folds 0/2 and 1/3."
        reason = "no_fixed_z_positive_proposal"
        if result is not None:
            assert result["proposed_rows"] == selected
            accepted_params = result["weight_params"] if result["weight_params"] is not None else parent_params
            accepted_changes = sparse_parameter_changes(parent_params, accepted_params)
            assert accepted_changes["changed_lv3_rows"] == result["accepted_rows"]
            report["fixed_z"].update(
                accepted_rows=result["accepted_rows"], parameters=accepted_changes,
                surviving_loss=loss_summary(result["before"], result["after"]),
            )
            assert len(capture["fixed_z_losses"]) == 2
            report["fixed_z"]["initial_proposal_bf16_loss"] = loss_summary(*capture["fixed_z_losses"])
            before, raw_after = capture["fixed_z_losses"]
            expected_take = (raw_after < before).all(0) & torch.isfinite(raw_after).all(0)
            assert int(expected_take.sum()) == result["accepted_rows"]
            assert_tree_equal(before, result["before"], "fixed_z.before")
            assert_tree_equal(torch.where(expected_take[None], raw_after, before), result["after"], "fixed_z.after")
            # Filtering must preserve every rejected output row exactly and keep
            # each accepted row identical to its initial joint discrete proposal.
            for name in parent_params:
                mask = expected_take.reshape(outputs, *([1]*(parent_params[name].ndim-1)))
                expected = torch.where(mask, proposed[name], parent_params[name])
                assert_tree_equal(expected, accepted_params[name], f"fixed_z.surviving.{name}")
            reason = "fixed_z_bf16_rejected" if result["accepted_rows"] == 0 else "state_recompile_rejected"
    if capture["recompilations"]:
        assert len(capture["recompilations"]) == 1
        rebuilt = capture["recompilations"][0]
        report["recompile"]["succeeded"] = rebuilt["result"] is not None
        if rebuilt["result"] is not None:
            report["recompile"]["state"] = compiled_state_changes(parent["activation_state"], rebuilt["result"], rebuilt["params"], module)
            reason = "encoder_guard_not_reached"
    if capture["guards"]:
        assert len(capture["guards"]) == 1
        guard = capture["guards"][0]
        report["encoder_guard"].update(
            accepted=guard["accept"], loss=loss_summary(guard["before"], guard["after"]),
            loss_units="Mean output MSE for even/odd rows of the full final calibration input.",
            relative_improvement_required=0.0001,
        )
        assert guard["accept"] == bool(torch.isfinite(guard["before"]).all()
                                      and torch.isfinite(guard["after"]).all()
                                      and (guard["after"] < guard["before"] * 0.9999).all())
        assert guard["accept"] == (not parent_fallback), "Final public result disagrees with encoder guard"
        assert len(guard_calls) == 2 and guard_calls[0]["shape"] == guard_calls[1]["shape"]
        reason = "accepted" if guard["accept"] else "encoder_guard_rejected"
    if capture["exceptions"]:
        assert parent_fallback, "Exception did not preserve exact parent fallback"
        reason = "exception_parent_fallback"
    report["outcome"] = reason
    return report


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
        "timing": "Serial public API wall time, excluding input/state cloning, checks, scoring and replay. Includes lightweight counters, reference capture, and nested pair/calibration timers; no extra tensor comparisons inside timed APIs. Retained candidate state can increase peak memory. Stage times overlap and must not be summed. No online-time conversion.",
        "validation_policy": "Independent calibrations with identical starting and ending RNG. Down is bitexact. Coordinate and teacher state, W scale_factor/scale_lv2 stay exact; each output row changes at most one lv3 microgroup. Up losses may rise and negative cases are retained. Dynamic input/state/RNG must remain unchanged.",
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
            instrumentation = {v: install_calibration_instrumentation(modules[v], required=v == candidate) for v in versions}
            learning_times = {v: instrumentation[v][0] for v in versions}
            captures = {v: instrumentation[v][1] for v in versions}
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
            assert item["calibration_end_rng_equal"], (gi, "Candidate added calibration RNG consumption")
            assert item["transform_comparison"]["bitexact"], (gi, "Frozen coordinate transform changed")
            audit_started = time.perf_counter()
            item["sparse_weight_audit"] = audit_sparse_calibration(
                modules[candidate], captures[candidate], calibrated[parent], calibrated[candidate],
            )
            item["timing"][str(candidate)]["post_calibration_audit"] = time.perf_counter() - audit_started
            for evidence in captures.values():
                evidence.clear()
            captures.clear()
            instrumentation.clear()
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
        report["sparse_weight_coverage"] = {
            "groups_probed": len(report["groups"]),
            "groups_with_shortlisted_groups": sum(g["sparse_weight_audit"]["proposals"]["shortlisted_groups"] > 0 for g in report["groups"]),
            "shortlisted_groups": sum(g["sparse_weight_audit"]["proposals"]["shortlisted_groups"] for g in report["groups"]),
            "proposed_rows": sum(g["sparse_weight_audit"]["proposals"]["selected_rows"] for g in report["groups"]),
            "fixed_z_accepted_rows": sum(g["sparse_weight_audit"]["fixed_z"]["accepted_rows"] for g in report["groups"]),
            "encoder_guard_calls": sum(g["sparse_weight_audit"]["encoder_guard"]["called"] for g in report["groups"]),
            "final_accepted_groups": sum(g["sparse_weight_audit"]["final_accepted"] for g in report["groups"]),
            "final_parent_fallback_groups": sum(g["sparse_weight_audit"]["parent_fallback"] for g in report["groups"]),
            "interpretation": "Shortlisting, FP32 fixed-Z proposals, BF16 fixed-Z survival, refreshed-encoder guard, and final public results are counted separately.",
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
    parser.add_argument("--candidate", type=int, default=425)
    parser.add_argument("--dataset", default="local_fitted_models_attention_feedback_v2")
    parser.add_argument("--groups", default="all", help="all or comma-separated zero-based group IDs")
    parser.add_argument("--output", default="docs/v425-linear-paired.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260907425)
    args = parser.parse_args()
    if args.threads < 1 or args.parent == args.candidate:
        parser.error("threads must be positive; parent and candidate must differ")
    run(args)


if __name__ == "__main__":
    main()
