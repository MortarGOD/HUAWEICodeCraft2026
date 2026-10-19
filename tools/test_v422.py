#!/usr/bin/env python3
"""Synthetic mechanism checks only; no local bank or official self-check run.

v422 changes the Linear coordinate transform, not its discrete search budgets.
The three-arm MSE report is diagnostic and does not select an implementation.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from self_check import validate_frozen_state, validate_hif4_params


def load(version: int, suffix: str = ""):
    spec = importlib.util.spec_from_file_location(
        f"v{version}_mechanism_{suffix}", ROOT / "solution" / f"solution_v{version}.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def assert_tree_equal(test, left, right):
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


def spiked_pair(blocks: int = 1):
    # Distinct activation and weight peaks: a W-only/A-only metric misses one.
    activation = torch.eye(64).repeat(2, 1)
    weight = torch.eye(64).repeat(2, 1)
    activation[:, 0] *= 50.0
    weight[:, 1] *= 40.0
    return activation.repeat(1, blocks), weight.repeat(1, blocks)


def joint_gram(activation, weight):
    ga = activation.T @ activation / activation.shape[0]
    gw = weight.T @ weight / weight.shape[0]
    return 0.5 * (ga / ga.diagonal().mean() + gw / gw.diagonal().mean())


class V422Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.parent = load(421)
        cls.child = load(422)

    def test_ast_scope_preserves_all_other_functions_and_imports(self):
        parent = ast.parse((ROOT / "solution/solution_v421.py").read_text())
        child = ast.parse((ROOT / "solution/solution_v422.py").read_text())
        retained = []
        names = []
        for node in child.body:
            if isinstance(node, ast.FunctionDef):
                if node.name in {"_v422_joint_block_transform", "_pair_polar_factorization"}:
                    names.append(node.name)
                    continue
                if node.name == "_v422_parent_pair_polar_factorization":
                    node.name = "_pair_polar_factorization"
            retained.append(node)
        self.assertEqual(set(names), {"_v422_joint_block_transform", "_pair_polar_factorization"})
        self.assertEqual(ast.dump(parent), ast.dump(ast.Module(body=retained, type_ignores=[])))
        forbidden = {
            "os", "pathlib", "io", "tempfile", "shutil", "subprocess", "socket",
            "ctypes", "pickle", "marshal", "importlib", "zipimport", "pkgutil",
            "runpy", "mmap", "resource", "pty", "tty", "sys", "builtins",
        }
        imports = set()
        for node in ast.walk(child):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        self.assertFalse(imports & forbidden)

    def test_joint_two_directions_spread_over_all_64_channels(self):
        activation, weight = spiked_pair()
        before = joint_gram(activation, weight)
        transform = self.child._v422_joint_block_transform(activation, weight)[0]
        after = transform.T @ before @ transform
        self.assertLess(float(after.diagonal().max() / before.diagonal().max()), 0.05)
        for row in (0, 1):
            torch.testing.assert_close(transform[row].abs(), torch.full((64,), 0.125), atol=0.002, rtol=0.0)
        # Eigh eigenvectors have arbitrary signs. The target is met up to
        # that irrelevant global sign, rather than a chosen input basis sign.
        constant = torch.full((64,), 0.125)
        alternating = torch.where(torch.arange(64) % 2 == 0, 0.125, -0.125)
        for row, target in ((0, constant), (1, alternating)):
            alignment = torch.sign(torch.dot(transform[row], target))
            torch.testing.assert_close(transform[row] * alignment, target, atol=0.002, rtol=0.0)
        # Frozen F need not be exactly orthogonal; the paired weight uses solve.
        orthogonality_error = float((transform @ transform.T - torch.eye(64)).abs().max())
        self.assertGreater(orthogonality_error, 0.0)
        self.assertLess(orthogonality_error, 0.02)
        print(json.dumps({"spectral_peak_before": float(before.diagonal().max()),
                          "spectral_peak_after_frozen_F": float(after.diagonal().max()),
                          "frozen_F_orthogonality_max_error": orthogonality_error}))

    def test_independent_a_w_scale_normalization(self):
        activation, weight = spiked_pair(2)
        first = self.child._v422_joint_block_transform(activation, weight)
        scaled = self.child._v422_joint_block_transform(activation * 8.0, weight * 0.25)
        self.assertTrue(torch.equal(first, scaled))
        one_side = self.child._v422_joint_block_transform(torch.zeros_like(activation), weight)
        self.assertFalse(torch.equal(one_side, torch.eye(64).repeat(2, 1, 1)))
        self.assertTrue(bool(torch.isfinite(one_side).all()))

    def test_isotropic_and_zero_spectra_are_identity(self):
        for activation, weight in ((torch.eye(64).repeat(2, 1), torch.eye(64)),
                                   (torch.zeros((7, 128)), torch.zeros((128, 128)))):
            got_a, got_w, transform = self.child._pair_polar_factorization(activation, weight)
            identity = torch.eye(64).repeat(weight.shape[1] // 64, 1, 1)
            self.assertTrue(torch.equal(transform, identity))
            self.assertTrue(torch.equal(got_a, activation))
            self.assertTrue(torch.equal(got_w, weight))

    def test_sample_threshold_can_leave_weak_direction_inactive(self):
        activation = torch.eye(64)
        weight = torch.eye(64)
        activation[:, 0] *= 2.0
        weight[:, 0] *= 2.0
        transform = self.child._v422_joint_block_transform(activation, weight)
        self.assertTrue(torch.equal(transform, torch.eye(64)[None]))

    def test_frozen_bf16_map_pairs_outputs_and_does_not_mutate_inputs_rng(self):
        generator = torch.Generator().manual_seed(422)
        for width in (64, 128):
            activation = torch.randn((39, width * 2), generator=generator)[:, ::2]
            weight = torch.randn((width + 17, width * 2), generator=generator)[:, ::2]
            activation[:, 0] *= 35
            weight[:, 1] *= 30
            before = copy.deepcopy((activation, weight))
            rng = torch.get_rng_state().clone()
            got_a, got_w, transform = self.child._pair_polar_factorization(activation, weight)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            assert_tree_equal(self, (activation, weight), before)
            self.assertEqual(tuple(transform.shape), (width // 64, 64, 64))
            self.assertTrue(torch.equal(transform, transform.bfloat16().float()))
            torch.testing.assert_close(got_a @ got_w.T, activation @ weight.T, atol=0.004, rtol=3e-5)
            actual_dynamic = torch.einsum("rgi,gij->rgj", activation.reshape(-1, width // 64, 64), transform.bfloat16().float()).reshape_as(activation)
            self.assertTrue(torch.equal(actual_dynamic, got_a))

    def test_unsupported_shape_uses_exact_parent_contract(self):
        generator = torch.Generator().manual_seed(4221)
        for shape, width in (((7, 32), 32), ((2, 7, 64), 64)):
            activation = torch.randn(shape, generator=generator)
            weight = torch.randn((64, width), generator=generator)
            expected = self.parent._pair_polar_factorization(activation, weight)
            actual = self.child._pair_polar_factorization(activation, weight)
            assert_tree_equal(self, actual, expected)
        sentinel = object()
        with patch.object(self.child, "_v422_parent_pair_polar_factorization", return_value=sentinel) as fallback:
            for activation, weight in ((torch.empty((0, 64)), torch.empty((64, 64))),
                                       (torch.empty((2, 0)), torch.empty((64, 0))),
                                       (torch.empty((2, 32)), torch.empty((64, 64)))):
                self.assertIs(self.child._pair_polar_factorization(activation, weight), sentinel)
            self.assertEqual(fallback.call_count, 3)

    def test_runtime_nonfinite_and_singular_fallback_once(self):
        activation, weight = spiked_pair()
        sentinel = object()
        for response in (None, torch.full((1, 64, 64), float("nan")), torch.zeros((1, 64, 64))):
            with patch.object(self.child, "_v422_joint_block_transform", return_value=response):
                with patch.object(self.child, "_v422_parent_pair_polar_factorization", return_value=sentinel) as fallback:
                    self.assertIs(self.child._pair_polar_factorization(activation, weight), sentinel)
                    fallback.assert_called_once_with(activation, weight)
        with patch.object(self.child, "_v422_joint_block_transform", side_effect=RuntimeError("synthetic eig failure")):
            with patch.object(self.child, "_v422_parent_pair_polar_factorization", return_value=sentinel) as fallback:
                self.assertIs(self.child._pair_polar_factorization(activation, weight), sentinel)
                fallback.assert_called_once()
        bad = activation.clone()
        bad[0, 0] = float("inf")
        with patch.object(self.child, "_v422_parent_pair_polar_factorization", return_value=sentinel) as fallback:
            with patch.object(self.child, "_v422_joint_block_transform") as helper:
                self.assertIs(self.child._pair_polar_factorization(bad, weight), sentinel)
                fallback.assert_called_once()
                helper.assert_not_called()
        # A programming error is not silently reclassified as a numeric fallback.
        with patch.object(self.child, "_v422_joint_block_transform", side_effect=ValueError("programming error")):
            with self.assertRaises(ValueError):
                self.child._pair_polar_factorization(activation, weight)

    def test_three_arm_real_legal_quantization_diagnostic(self):
        generator = torch.Generator().manual_seed(4222)
        activation = torch.randn((128, 64), generator=generator)
        weight = torch.randn((128, 64), generator=generator)
        activation[:, 0] *= 20
        weight[:, 1] *= 15
        target = activation @ weight.T
        arms = {"identity": (activation, weight, torch.eye(64)[None]),
                "parent_polar": self.parent._pair_polar_factorization(activation, weight),
                "joint_reflection": self.child._pair_polar_factorization(activation, weight)}
        records = {}
        params = {}
        for name, (a, w, transform) in arms.items():
            qa = self.child._quantize_linear_values(a)
            qw = self.child._quantize_linear_values(w)
            self.assertEqual(validate_hif4_params(qa, a.shape, name), [])
            self.assertEqual(validate_hif4_params(qw, w.shape, name), [])
            decoded_a = self.child._linear_score_decode(qa)
            decoded_w = self.child._linear_score_decode(qw)
            output = (decoded_a @ decoded_w.T).float()
            records[name] = {"output_mse": float((target - output).square().mean()),
                             "continuous_output_mse": float((target - a @ w.T).square().mean())}
            params[name] = (qa, qw, decoded_a, decoded_w)
        identity, reflected = params["identity"], params["joint_reflection"]
        sign_mant_changed = sum(int((identity[side][key] != reflected[side][key]).sum())
                                for side in (0, 1) for key in ("sign", "mant"))
        decoded_changed = sum(int((identity[side] != reflected[side]).sum()) for side in (2, 3))
        self.assertGreater(sign_mant_changed, 0)
        self.assertGreater(decoded_changed, 0)
        print(json.dumps({"three_arm_synthetic_diagnostic": records,
                          "sign_mant_changed_vs_identity": sign_mant_changed,
                          "decoded_values_changed_vs_identity": decoded_changed}))

    def test_small_calibration_state_and_fresh_dynamic_module(self):
        generator = torch.Generator().manual_seed(4223)
        pattern = (2 * torch.randint(0, 2, (1, 64), generator=generator) - 1).bfloat16()
        def packed(rows):
            quant = pattern.repeat(rows, 1)
            noise = torch.randint(0, 8, (rows, 64), generator=generator) == 0
            quant = torch.where(noise, -quant, quant).contiguous()
            return quant, torch.ones((rows, 4), dtype=torch.bfloat16)
        weight = packed(128)
        calibration = [packed(8), packed(8)]
        frozen = copy.deepcopy((weight, calibration))
        result = self.child.hif4_calibration_and_quantize_weight(*weight, calibration)
        assert_tree_equal(self, (weight, calibration), frozen)
        state = result["activation_state"]
        self.assertEqual(validate_frozen_state(state, "v422-state"), [])
        self.assertEqual(tuple(state["unrestricted_pair_transform"].shape), (1, 64, 64))
        self.assertEqual(state["unrestricted_pair_transform"].dtype, torch.bfloat16)
        self.assertEqual(validate_hif4_params(result["weight_params"], weight[0].shape, "v422-weight"), [])
        activation = packed(3)
        frozen_state_input = copy.deepcopy((state, activation))
        rng = torch.get_rng_state().clone()
        original = self.child.hif4_dynamic_quantize_activation(*activation, state)
        fresh = load(422, "fresh")
        copied_state = copy.deepcopy(state)
        independent = fresh.hif4_dynamic_quantize_activation(*activation, copied_state)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        assert_tree_equal(self, (state, activation), frozen_state_input)
        assert_tree_equal(self, copied_state, state)
        assert_tree_equal(self, original, independent)
        self.assertEqual(validate_hif4_params(independent, activation[0].shape, "fresh-v422"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
