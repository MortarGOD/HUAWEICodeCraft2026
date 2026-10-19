"""Bitwise regression for Linear initial-candidate preparation reuse.

Run with the repository Torch runtime, for example:
    /opt/miniconda3/envs/codecraft/bin/python tools/test_v420.py
    /opt/miniconda3/envs/codecraft/bin/python tools/test_v420.py --self-check

Only a single real Linear group is calibrated by default. The broad layout and
rounding matrix exercises the quantizer directly, not the expensive complete
target solver. No candidate search budget is changed in the real-data check.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import random
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_local import load_solution
from self_check import (
    _normalize_linear_group,
    self_check,
    validate_hif4_params,
    validate_linear_calibration_result,
)


def assert_exact(actual: Any, expected: Any, label: str = "result") -> None:
    """Require dtype/shape and raw tensor bits, including signed zeros."""
    assert type(actual) is type(expected), (label, type(actual), type(expected))
    if isinstance(actual, torch.Tensor):
        assert actual.dtype == expected.dtype, (label, actual.dtype, expected.dtype)
        assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
        assert actual.device == expected.device, (label, actual.device, expected.device)
        assert actual.layout == expected.layout == torch.strided, label
        left = actual.detach().contiguous().reshape(-1).view(torch.uint8)
        right = expected.detach().contiguous().reshape(-1).view(torch.uint8)
        assert torch.equal(left, right), f"{label}: tensor bytes differ"
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys(), (label, actual.keys(), expected.keys())
        for key in actual:
            assert_exact(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected), label
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_exact(left, right, f"{label}[{index}]")
    elif isinstance(actual, float):
        assert struct.pack("!d", actual) == struct.pack("!d", expected), label
    else:
        assert actual == expected, (label, actual, expected)


def rng_snapshot() -> tuple[torch.Tensor, tuple]:
    return torch.random.get_rng_state().clone(), random.getstate()


def restore_rng(snapshot: tuple[torch.Tensor, tuple]) -> None:
    torch.random.set_rng_state(snapshot[0])
    random.setstate(snapshot[1])


def rng_equal(left: tuple[torch.Tensor, tuple], right: tuple[torch.Tensor, tuple]) -> bool:
    return torch.equal(left[0], right[0]) and left[1] == right[1]


def ast_scope_check(parent_path: Path, candidate_path: Path) -> dict[str, Any]:
    parent = ast.parse(parent_path.read_text())
    candidate = ast.parse(candidate_path.read_text())
    dump = lambda node: ast.dump(node, include_attributes=False)
    assert len(parent.body) == len(candidate.body), "module statements added/removed"
    changes = [
        (old, new)
        for old, new in zip(parent.body, candidate.body)
        if dump(old) != dump(new)
    ]
    assert len(changes) == 1, f"expected one changed definition, got {len(changes)}"
    old, new = changes[0]
    name = "_quantize_linear_exact_target"
    assert isinstance(old, ast.FunctionDef) and isinstance(new, ast.FunctionDef)
    assert old.name == new.name == name, (getattr(old, "name", None), new.name)
    assert dump(old.args) == dump(new.args), "target signature changed"
    assert [dump(n) for n in old.decorator_list] == [dump(n) for n in new.decorator_list]
    assert dump(old.returns) == dump(new.returns), "target return annotation changed"

    def assignment_index(body: list[ast.stmt], target: str) -> int:
        found = [
            index for index, node in enumerate(body)
            if isinstance(node, ast.Assign)
            and any(isinstance(item, ast.Name) and item.id == target for item in node.targets)
        ]
        assert len(found) == 1, (name, target, found)
        return found[0]

    old_begin = assignment_index(old.body, "centers")
    new_begin = assignment_index(new.body, "centers")
    old_end = assignment_index(old.body, "reconstructed")
    new_end = assignment_index(new.body, "reconstructed")
    assert [dump(n) for n in old.body[:old_begin]] == [dump(n) for n in new.body[:new_begin]], \
        "candidate specifications or earlier target logic changed"
    assert [dump(n) for n in old.body[old_end:]] == [dump(n) for n in new.body[new_end:]], \
        "target solver after initial candidate generation changed"
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_quantize_linear_values"
        and any(keyword.arg == "prepared" for keyword in node.keywords)
        for statement in new.body[new_begin:new_end]
        for node in ast.walk(statement)
    ), "initial candidates do not pass prepared"
    return {
        "changed_definition": name,
        "initial_candidate_region_only": True,
        "other_module_statements_identical": len(parent.body) - 1,
        "parent_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
    }


def prepare(module: Any, values: torch.Tensor, importance: torch.Tensor | None = None):
    """Construct the existing public helper's prepared tuple in isolation."""
    channels = int(values.shape[-1])
    groups = values.to(torch.float32).reshape(*values.shape[:-1], channels // 64, 8, 2, 4)
    weights = None if importance is None else importance.to(torch.float32).reshape(
        1, channels // 64, 8, 2, 4
    )
    peak = groups.abs().amax(dim=(-3, -2, -1), keepdim=True) / 7.0
    return groups, weights, module._encode_e6m2(peak)


def quantizer_cases(generator: torch.Generator):
    for dtype in (torch.float32, torch.bfloat16):
        # Different prefix ranks, a transposed matrix, and a strided channel axis.
        cases = [
            ("vector", torch.randn(64, generator=generator).to(dtype)),
            ("strided-vector", torch.randn(128, generator=generator).to(dtype)[::2]),
            ("transposed", torch.randn(128, 3, generator=generator).to(dtype).T),
            ("prefix2-strided", torch.randn(2, 3, 256, generator=generator).to(dtype)[..., ::2]),
            ("prefix3", torch.randn(2, 1, 3, 64, generator=generator).to(dtype)),
            ("zero", torch.zeros(2, 128, dtype=dtype)),
        ]
        # Finite exact HiF4 endpoints, smallest mantissa step, signed zero and
        # tiny values. These are helper inputs, not a claim about an NVFP4 scale.
        endpoints = torch.tensor(
            [-344064.0, 344064.0, -(2.0 ** -50), 2.0 ** -50,
             -0.0, 0.0, -(7.0 * 2.0 ** -48), 7.0 * 2.0 ** -48],
            dtype=dtype,
        ).repeat(16).reshape(2, 64)
        cases.append(("finite-endpoints", endpoints))
        for name, values in cases:
            yield f"{str(dtype).split('.')[-1]}/{name}", values


def prepared_equivalence(parent: Any, candidate: Any) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(420)
    configurations = (
        ((1,), 0.5),
        ((-2, 0, 1, 2), 0.375),
        ((0, 1, 1, 2, 7), 0.625),
        ((0, 1, 2), 0.5),
    )
    comparisons = 0
    cases_seen = []
    saved_rng = rng_snapshot()
    for case_name, values in quantizer_cases(generator):
        cases_seen.append(case_name)
        channels = int(values.shape[-1])
        modes = [("plain", None, None, None)]
        if values.ndim >= 2:
            importance = torch.rand(channels, generator=generator).add(0.05)
            importance[::7] = 0
            projection = torch.randn(2, channels // 64, 8, 2, 4, generator=generator) * 0.05
            strength = torch.rand(2, channels // 64, generator=generator) * 0.1
            modes.append(("weighted-projected", importance, projection, strength))
        for mode, importance, projection, strength in modes:
            prepared = prepare(parent, values, importance)
            before = copy.deepcopy((values, importance, projection, strength, prepared))
            for offsets, threshold in configurations:
                label = f"{case_name}/{mode}/{offsets}/{threshold}"
                kwargs = dict(
                    importance=importance,
                    offsets=offsets,
                    projection=projection,
                    projection_strength=strength,
                    rounding_threshold=threshold,
                )
                expected = parent._quantize_linear_values(values, **kwargs)
                for name, module, argument in (
                    ("parent-prepared", parent, prepared),
                    ("candidate-unprepared", candidate, None),
                    ("candidate-prepared", candidate, prepared),
                ):
                    result = module._quantize_linear_values(values, prepared=argument, **kwargs)
                    assert_exact(result, expected, f"{label}/{name}")
                    comparisons += 1
                assert not validate_hif4_params(expected, values.shape, label), label
                assert_exact((values, importance, projection, strength, prepared), before, f"{label}/inputs")
    assert rng_equal(saved_rng, rng_snapshot()), "direct quantizer changed RNG"
    return {"cases": cases_seen, "bitwise_comparisons": comparisons, "inputs_and_rng_unchanged": True}


def target_fixture(generator: torch.Generator, prefix: tuple[int, ...], *, alternate: bool,
                   full: bool, extra: bool, wide: bool, task_rounding: bool):
    width, outputs = 128, 160
    values = torch.randn(*prefix, width, generator=generator) * 0.4
    # Keep the channel axis non-contiguous in every complete solver fixture.
    storage = torch.empty(*prefix, width * 2)
    storage[..., ::2] = values
    values = storage[..., ::2]
    alternate_values = values * 1.375 + 0.025 if alternate else None
    deployed = torch.randn(outputs, width, generator=generator) * 0.15
    target = values @ deployed.T + torch.randn(*prefix, outputs, generator=generator) * 0.01
    gram = deployed.T @ deployed
    block_gram = torch.stack([gram[start:start + 64, start:start + 64] for start in (0, 64)])
    block_inverse = torch.linalg.inv(block_gram + torch.eye(64) * 0.02)
    micro_gram = torch.stack([
        torch.stack([block[start:start + 4, start:start + 4] for start in range(0, 64, 4)])
        for block in block_gram
    ])
    micro_inverse = torch.linalg.inv(micro_gram + torch.eye(4) * 0.02)
    return dict(
        values=values,
        target=target,
        deployed_weight=deployed,
        block_gram=block_gram,
        block_inverse=block_inverse,
        micro_inverse=micro_inverse,
        super_gram=gram.unsqueeze(0) if extra else None,
        super_inverse=torch.linalg.inv(gram + torch.eye(width) * 0.02).unsqueeze(0) if extra else None,
        scale_offsets=(0, 1, 2, 6, 7),
        task_rounding=task_rounding,
        wide_candidates=wide,
        extra_refinement=extra,
        search_sweeps_override=1,
        refinement_sweeps_override=1,
        micro_passes_override=1,
        alternate_values=alternate_values,
        full_gram=gram if full else None,
    )


def observed_target(module: Any, fixture: dict[str, Any], *, expect_prepared: bool):
    """Observe the real initial prefix and let the unmodified solver finish."""
    centers = [fixture["values"]]
    if fixture["alternate_values"] is not None:
        centers.append(fixture["alternate_values"])
    expected_prepared = [prepare(module, center) for center in centers]
    original_quantizer = module._quantize_linear_values
    original_encoder = module._encode_e6m2
    records = []
    preparation_refs = {}
    preparation_snapshots = {}
    encoder_calls = 0
    after_initial_encoder_calls = 0
    later_quantizer_calls = 0

    def encoder(*args, **kwargs):
        nonlocal encoder_calls
        encoder_calls += 1
        return original_encoder(*args, **kwargs)

    def quantizer(values, *args, **kwargs):
        nonlocal after_initial_encoder_calls, later_quantizer_calls
        center_index = next((index for index, center in enumerate(centers) if values is center), None)
        start_calls = encoder_calls
        if center_index is not None:
            # Initial target candidates pass these arguments by keyword.
            assert not args, "initial quantizer call convention changed"
            prepared = kwargs.get("prepared")
            assert (prepared is not None) == expect_prepared, "unexpected initial prepared policy"
            if expect_prepared:
                assert_exact(prepared, expected_prepared[center_index], "center preparation")
                if center_index not in preparation_refs:
                    preparation_refs[center_index] = prepared
                    preparation_snapshots[center_index] = copy.deepcopy(prepared)
                else:
                    assert prepared is preparation_refs[center_index], "prepared tuple was rebuilt for a specification"
                    assert all(left is right for left, right in zip(prepared, preparation_refs[center_index])), \
                        "prepared components were rebuilt"
        else:
            later_quantizer_calls += 1
        result = original_quantizer(values, *args, **kwargs)
        if center_index is not None:
            assert encoder_calls - start_calls == (0 if expect_prepared else 1), \
                "unexpected per-candidate encode count"
            records.append((center_index, tuple(kwargs["offsets"]), kwargs["rounding_threshold"], copy.deepcopy(result)))
            after_initial_encoder_calls = encoder_calls
        return result

    module._quantize_linear_values = quantizer
    module._encode_e6m2 = encoder
    before = copy.deepcopy(fixture)
    rng_before = rng_snapshot()
    try:
        result = module._quantize_linear_exact_target(**fixture)
    finally:
        module._quantize_linear_values = original_quantizer
        module._encode_e6m2 = original_encoder
    assert records and later_quantizer_calls, "complete target path was not exercised"
    assert_exact(fixture, before, "target fixture inputs")
    assert rng_equal(rng_before, rng_snapshot()), "target solver changed RNG"
    if expect_prepared:
        assert len(preparation_refs) == len(centers)
        assert len({id(value) for value in preparation_refs.values()}) == len(centers), \
            "different centers share a prepared tuple"
        assert len({id(value[0]) for value in preparation_refs.values()}) == len(centers), \
            "different centers share grouped values"
        assert after_initial_encoder_calls == len(centers), "preparation was not encoded exactly once per center"
        for index in preparation_refs:
            assert_exact(preparation_refs[index], preparation_snapshots[index], "prepared input mutation")
    else:
        assert after_initial_encoder_calls == len(records)
    return result, records, {
        "centers": len(centers),
        "initial_candidates": len(records),
        "initial_peak_encodes": after_initial_encoder_calls,
        "later_quantizer_calls": later_quantizer_calls,
    }


def target_regression(parent: Any, candidate: Any) -> list[dict[str, Any]]:
    generator = torch.Generator().manual_seed(421)
    modes = (
        dict(prefix=(3,), alternate=False, full=False, extra=False, wide=False, task_rounding=False),
        dict(prefix=(2, 3), alternate=True, full=False, extra=False, wide=True, task_rounding=True),
        dict(prefix=(3,), alternate=True, full=True, extra=True, wide=True, task_rounding=True),
    )
    summaries = []
    for mode in modes:
        fixture = target_fixture(generator, **mode)
        expected, old_records, old_counts = observed_target(parent, fixture, expect_prepared=False)
        actual, new_records, new_counts = observed_target(candidate, fixture, expect_prepared=True)
        assert_exact(new_records, old_records, "initial candidate order/parameters")
        assert_exact(actual, expected, "complete target solver")
        assert not validate_hif4_params(actual, fixture["values"].shape, "target")
        summaries.append(dict(mode=mode, parent=old_counts, candidate=new_counts))
    return summaries


def real_linear_regression(parent: Any, candidate: Any, candidate_path: Path,
                           dataset: Path, group_index: int) -> dict[str, Any]:
    raw = torch.load(dataset, map_location="cpu", weights_only=False)
    group = _normalize_linear_group(raw[group_index], group_index)
    del raw
    weight = (group["weight_quant"], group["weight_scale"])
    calibration = group["calib_activation_list"]
    tests = group["test_activation_list"]
    fresh = load_solution(candidate_path)
    before_inputs = copy.deepcopy(group)
    initial_rng = rng_snapshot()
    states = []
    rng_afters = []
    target_calls = [0, 0]
    try:
        for index, module in enumerate((parent, candidate)):
            restore_rng(initial_rng)
            original = module._quantize_linear_exact_target

            def counted(*args, _index=index, _original=original, **kwargs):
                target_calls[_index] += 1
                return _original(*args, **kwargs)

            module._quantize_linear_exact_target = counted
            try:
                result = module.hif4_calibration_and_quantize_weight(*weight, calibration)
            finally:
                module._quantize_linear_exact_target = original
            assert not validate_linear_calibration_result(result, weight[0].shape, "real calibration")
            assert_exact(group, before_inputs, "calibration inputs")
            states.append(result)
            rng_afters.append(rng_snapshot())
        assert_exact(states[1], states[0], "real calibration weight/state")
        assert rng_equal(rng_afters[1], rng_afters[0]), "calibration RNG behavior differs from parent"
        assert target_calls[0] == target_calls[1], "real calibration target invocation count changed"
        # Some historical parent calibration paths reset the global RNG. Record
        # this inherited behavior; direct quantizer and dynamic calls below must
        # leave it untouched. The test restores its caller RNG on exit as well.
        calibration_rng_changed = not rng_equal(initial_rng, rng_afters[0])
        restore_rng(initial_rng)
        snapshots = copy.deepcopy(states)
        fresh_state = copy.deepcopy(states[1]["activation_state"])
        fresh_snapshot = copy.deepcopy(fresh_state)
        for index, packed in enumerate(tests):
            outputs = []
            for module, state in (
                (parent, states[0]["activation_state"]),
                (candidate, states[1]["activation_state"]),
                (fresh, fresh_state),
            ):
                rng_before = rng_snapshot()
                output = module.hif4_dynamic_quantize_activation(*packed, state)
                assert rng_equal(rng_before, rng_snapshot()), f"dynamic {index} changed RNG"
                assert not validate_hif4_params(output, packed[0].shape, f"real activation {index}")
                outputs.append(output)
            assert_exact(outputs[1], outputs[0], f"real activation {index}/candidate")
            assert_exact(outputs[2], outputs[0], f"real activation {index}/fresh module")
            assert_exact(states, snapshots, f"real activation {index}/states")
            assert_exact(fresh_state, fresh_snapshot, f"real activation {index}/fresh state")
            assert_exact(group, before_inputs, f"real activation {index}/inputs")
        return {
            "dataset": str(dataset),
            "group": group_index,
            "weight_shape": list(weight[0].shape),
            "calibration_shapes": [list(pair[0].shape) for pair in calibration],
            "test_shapes": [list(pair[0].shape) for pair in tests],
            "target_calls_during_calibration": target_calls,
            "parent_calibration_changed_rng": calibration_rng_changed,
            "candidate_calibration_rng_matches_parent": True,
            "weight_state_and_activations_bitwise_equal": True,
            "inputs_states_and_dynamic_rng_unchanged": True,
            "fresh_module_accepts_frozen_state": True,
        }
    finally:
        restore_rng(initial_rng)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-version", type=int, default=419)
    parser.add_argument("--candidate-version", type=int, default=420)
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "local_fitted_models_attention_feedback_v2" / "linear.pt")
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--static-only", action="store_true", help="only check AST scope and source hashes")
    parser.add_argument("--skip-real", action="store_true", help="skip the complete real-data calibration check")
    parser.add_argument("--self-check", action="store_true", help="also run the official mini_sample self_check")
    args = parser.parse_args()
    if args.static_only and args.self_check:
        parser.error("--static-only cannot be combined with --self-check")
    if args.threads < 1 or args.group < 0:
        parser.error("--threads must be positive and --group nonnegative")
    return args


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    parent_path = ROOT / "solution" / f"solution_v{args.parent_version}.py"
    candidate_path = ROOT / "solution" / f"solution_v{args.candidate_version}.py"
    report = {
        "parent_version": args.parent_version,
        "candidate_version": args.candidate_version,
        "scope": ast_scope_check(parent_path, candidate_path),
    }
    if args.static_only:
        report["runtime_checks"] = "not run"
    else:
        torch.set_num_threads(args.threads)
        parent = load_solution(parent_path)
        candidate = load_solution(candidate_path)
        report["prepared"] = prepared_equivalence(parent, candidate)
        report["target"] = target_regression(parent, candidate)
        report["real_linear"] = "skipped" if args.skip_real else real_linear_regression(
            parent, candidate, candidate_path, args.dataset.resolve(), args.group
        )
        if args.self_check:
            with tempfile.TemporaryDirectory(prefix="v420-self-check-") as directory:
                (Path(directory) / "solution.py").symlink_to(candidate_path)
                assert self_check(directory), "official self_check failed"
            report["official_self_check"] = "passed"
        else:
            report["official_self_check"] = "not requested"
    report["passed_requested_checks"] = True
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
