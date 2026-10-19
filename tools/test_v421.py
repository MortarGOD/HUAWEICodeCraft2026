#!/usr/bin/env python3
"""Small synthetic mechanism and contract checks for the v421 Linear suffix."""

from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from self_check import validate_frozen_state, validate_hif4_params


def load(version: int, suffix: str = ""):
    path = ROOT / "solution" / f"solution_v{version}.py"
    spec = importlib.util.spec_from_file_location(f"test_v{version}_{suffix}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def params_from_codes(codes: torch.Tensor):
    rows, width = codes.shape
    blocks = width // 64
    return {
        "scale_factor": torch.ones((rows, blocks, 1, 1, 1), dtype=torch.bfloat16),
        "scale_lv2": torch.ones((rows, blocks, 8, 1, 1), dtype=torch.bfloat16),
        "scale_lv3": torch.ones((rows, blocks, 8, 2, 1), dtype=torch.bfloat16),
        "sign": torch.sign(codes).reshape(rows, blocks, 8, 2, 4).to(torch.bfloat16),
        "mant": (codes.abs() * 0.25).reshape(rows, blocks, 8, 2, 4).to(torch.bfloat16),
    }


def assert_tree_equal(test: unittest.TestCase, left, right):
    test.assertIs(type(left), type(right))
    if isinstance(left, torch.Tensor):
        test.assertEqual(left.dtype, right.dtype)
        test.assertEqual(left.shape, right.shape)
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            assert_tree_equal(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_tree_equal(test, a, b)
    else:
        test.assertEqual(left, right)


def state_for(weight: torch.Tensor):
    weight = weight.to(torch.bfloat16).contiguous()
    return {
        "v421_prefix_enabled": True,
        "v421_teacher_weight": weight.clone(),
        "unrestricted_deployed_weight": weight.clone(),
        "linear_full_gram": weight.float().T @ weight.float(),
    }


class V421Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.parent = load(420)
        cls.child = load(421)

    def test_full_v420_prefix_and_import_boundary(self):
        parent = (ROOT / "solution/solution_v420.py").read_bytes()
        child = (ROOT / "solution/solution_v421.py").read_bytes()
        self.assertTrue(child.startswith(parent))
        forbidden = {
            "os", "pathlib", "io", "tempfile", "shutil", "subprocess", "socket",
            "ctypes", "pickle", "marshal", "importlib", "zipimport", "pkgutil",
            "runpy", "mmap", "resource", "pty", "tty", "sys", "builtins",
        }
        parsed = ast.parse(child.decode())
        imports = set()
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        self.assertFalse(imports & forbidden)
        suffix = ast.parse(child[len(parent):].decode())
        names = [node.name for node in suffix.body if isinstance(node, ast.FunctionDef)]
        self.assertFalse(any("attention" in name or name.endswith(("_q", "_k", "_v")) for name in names))

    def test_prefix_cost_matches_explicit_output_for_every_prefix(self):
        weight = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
        residual = torch.tensor([[0.2, 0.3], [0.4, -0.1]])
        channels = torch.tensor([[0, 1, 2], [2, 0, 1]])
        deltas = torch.tensor([[0.25, -0.25, 0.5], [-0.5, 0.25, 0.25]])
        gradient = torch.gather(residual @ weight, 1, channels)
        predicted = self.child._v421_prefix_changes(gradient, deltas, channels, weight.T @ weight)
        for row in range(2):
            for prefix in range(1, 4):
                change = sum(deltas[row, j] * weight[:, channels[row, j]] for j in range(prefix))
                explicit = (residual[row] - change).square().sum() - residual[row].square().sum()
                torch.testing.assert_close(predicted[row, prefix - 1], explicit, atol=2e-7, rtol=1e-6)

    def test_cross_block_prefix_and_actual_teacher_no_regression(self):
        # Row 0: individually good equal columns must not both be accepted.
        # Row 1: a useful two-coordinate update spans separate 64-wide blocks.
        weight = torch.zeros((128, 128))
        weight[0, 0] = weight[0, 64] = 1.0
        weight[1, 1] = weight[2, 65] = 1.0
        values = torch.zeros((3, 128), dtype=torch.bfloat16)
        values[0, 0] = 0.5
        values[1, 1] = values[1, 65] = 0.5
        params = params_from_codes(torch.zeros((3, 128)))
        state = state_for(weight)
        frozen = copy.deepcopy((params, values, state))
        rng = torch.get_rng_state().clone()
        result = self.child._v421_refine_linear_prefix(params, values, state)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        assert_tree_equal(self, (params, values, state), frozen)
        for key in ("scale_factor", "scale_lv2", "scale_lv3"):
            self.assertIs(result[key], params[key])
        decoded = self.child._linear_score_decode(result).reshape(3, 128)
        self.assertEqual(int(torch.count_nonzero(decoded[0])), 1)
        self.assertEqual(int(torch.count_nonzero(decoded[1, :64])), 1)
        self.assertEqual(int(torch.count_nonzero(decoded[1, 64:])), 1)
        self.assertTrue(torch.equal(decoded[2], torch.zeros_like(decoded[2])))
        target = values.float() @ state["v421_teacher_weight"].float().T
        before = self.child._linear_score_decode(params).reshape(3, 128) @ state["unrestricted_deployed_weight"].T
        after = decoded @ state["unrestricted_deployed_weight"].T
        old_loss = (target - before.float()).square().mean(-1)
        new_loss = (target - after.float()).square().mean(-1)
        self.assertTrue(bool((new_loss <= old_loss).all()))
        self.assertTrue(bool((new_loss[:2] < old_loss[:2]).all()))
        self.assertEqual(validate_hif4_params(result, values.shape, "v421"), [])
        self.assertEqual(validate_frozen_state(state, "v421"), [])

    def test_bf16_gemm_tie_rejects_a_fp32_prediction_win(self):
        weight = torch.zeros((128, 128))
        weight[0, 0], weight[0, 64] = 256.0, 1.0
        values = torch.zeros((1, 128), dtype=torch.bfloat16)
        values[0, 0], values[0, 64] = 1.0, 0.25
        codes = torch.zeros((1, 128))
        codes[0, 0] = 4.0
        params = params_from_codes(codes)
        state = state_for(weight)
        observed = []
        original = self.child._v421_prefix_changes

        def capture(*args):
            changes = original(*args)
            observed.append(changes)
            return changes

        with patch.object(self.child, "_v421_prefix_changes", side_effect=capture):
            result = self.child._v421_refine_linear_prefix(params, values, state)
        self.assertTrue(bool((observed[0] < 0).any()))
        self.assertIs(result, params)

    def test_zero_unmarked_large_rows_and_nonmatrix_fall_back(self):
        state = state_for(torch.eye(128))
        params = params_from_codes(torch.zeros((1, 128)))
        zero = torch.zeros((1, 128), dtype=torch.bfloat16)
        self.assertIs(self.child._v421_refine_linear_prefix(params, zero, state), params)
        empty = torch.empty((0, 128), dtype=torch.bfloat16)
        self.assertIs(self.child._v421_refine_linear_prefix(params, empty, state), params)
        self.assertIs(self.child._v421_refine_linear_prefix(params, zero, {}), params)
        direct = dict(state, direct_output=True)
        self.assertIs(self.child._v421_refine_linear_prefix(params, zero, direct), params)
        for shape in ((129, 128), (1, 1, 128)):
            values = torch.zeros(shape, dtype=torch.bfloat16)
            self.assertIs(self.child._v421_refine_linear_prefix(params, values, state), params)
            scales = torch.ones((*shape[:-1], 8), dtype=torch.bfloat16)
            with patch.object(self.child, "_v421_parent_activation", return_value=params) as parent:
                with patch.object(self.child, "_v421_refine_linear_prefix") as refine:
                    self.assertIs(self.child.hif4_dynamic_quantize_activation(values, scales, state), params)
                    parent.assert_called_once()
                    refine.assert_not_called()

    def test_marker_added_only_after_parent_calibration_and_decoder_check(self):
        codes = torch.full((128, 64), 4.0)
        weight_params = params_from_codes(codes)
        deployed = self.child._linear_score_decode(weight_params)
        state = {"unrestricted_deployed_weight": deployed, "linear_full_gram": deployed.float().T @ deployed.float()}
        parent_result = {"weight_params": weight_params, "activation_state": state}
        frozen = copy.deepcopy(parent_result)
        quant = torch.ones((128, 64), dtype=torch.bfloat16)
        scale = torch.ones((128, 4), dtype=torch.bfloat16)
        fake_activation = params_from_codes(torch.zeros((1, 64)))

        def calibrate(*args):
            # This global lookup mirrors the parent joint-decoder calibration.
            self.child.hif4_dynamic_quantize_activation(quant[:1], scale[:1], state)
            return parent_result

        with patch.object(self.child, "_v421_parent_calibration", side_effect=calibrate):
            with patch.object(self.child, "_v421_parent_activation", return_value=fake_activation):
                with patch.object(self.child, "_v421_refine_linear_prefix") as refine:
                    result = self.child.hif4_calibration_and_quantize_weight(quant, scale, [])
                    refine.assert_not_called()
        assert_tree_equal(self, parent_result, frozen)
        self.assertTrue(result["activation_state"]["v421_prefix_enabled"])
        teacher = result["activation_state"]["v421_teacher_weight"]
        self.assertEqual(teacher.dtype, torch.bfloat16)
        self.assertEqual(teacher.numel() * teacher.element_size(), 2 * 128 * 64)
        stripped = dict(result["activation_state"])
        stripped.pop("v421_prefix_enabled")
        stripped.pop("v421_teacher_weight")
        assert_tree_equal(self, stripped, state)
        for replacement in (
            dict(state, unrestricted_deployed_weight=torch.zeros_like(deployed)),
            dict(state, direct_output=True),
            {"unrestricted_deployed_weight": deployed},
        ):
            rejected = dict(parent_result, activation_state=replacement)
            with patch.object(self.child, "_v421_parent_calibration", return_value=rejected):
                self.assertIs(self.child.hif4_calibration_and_quantize_weight(quant, scale, []), rejected)

    def test_small_real_parent_calibration_and_fresh_module(self):
        generator = torch.Generator().manual_seed(421)

        def packed(rows):
            choices = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.bfloat16)
            quant = choices[torch.randint(0, 8, (rows, 64), generator=generator)]
            signs = 2 * torch.randint(0, 2, (rows, 64), generator=generator) - 1
            return (quant * signs.to(torch.bfloat16)).contiguous(), torch.ones((rows, 4), dtype=torch.bfloat16)

        weight = packed(128)
        calibration = [packed(2), packed(2)]
        frozen_inputs = copy.deepcopy((weight, calibration))
        initial_rng = torch.get_rng_state().clone()
        parent = self.parent.hif4_calibration_and_quantize_weight(*weight, calibration)
        parent_rng = torch.get_rng_state().clone()
        torch.set_rng_state(initial_rng)
        child = self.child.hif4_calibration_and_quantize_weight(*weight, calibration)
        # The parent calibration itself consumes RNG; v421 must add no change
        # to that existing effect or to its chosen weights/statistics.
        self.assertTrue(torch.equal(parent_rng, torch.get_rng_state()))
        assert_tree_equal(self, (weight, calibration), frozen_inputs)
        assert_tree_equal(self, child["weight_params"], parent["weight_params"])
        stripped = dict(child["activation_state"])
        self.assertTrue(stripped.pop("v421_prefix_enabled"))
        stripped.pop("v421_teacher_weight")
        assert_tree_equal(self, stripped, parent["activation_state"])
        self.assertEqual(validate_frozen_state(child["activation_state"], "state"), [])
        fresh = load(421, "fresh")
        activation = packed(2)
        frozen = copy.deepcopy((activation, child["activation_state"]))
        rng = torch.get_rng_state().clone()
        result = fresh.hif4_dynamic_quantize_activation(*activation, child["activation_state"])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        assert_tree_equal(self, (activation, child["activation_state"]), frozen)
        self.assertEqual(validate_hif4_params(result, (2, 64), "fresh-v421"), [])
        before = self.parent.hif4_dynamic_quantize_activation(*activation, parent["activation_state"])
        target = fresh.dequantize_nvfp4(*activation).float() @ fresh.dequantize_nvfp4(*weight).float().T
        deployed = fresh._linear_score_decode(child["weight_params"])
        before_loss = (target - (fresh._linear_score_decode(before) @ deployed.T).float()).square().mean(-1)
        after_loss = (target - (fresh._linear_score_decode(result) @ deployed.T).float()).square().mean(-1)
        self.assertTrue(bool((after_loss <= before_loss).all()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
