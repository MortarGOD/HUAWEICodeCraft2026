#!/usr/bin/env python3
"""Synthetic v423 mechanism checks, separate from official/local-bank evaluation."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.test_v421 import assert_tree_equal, load, params_from_codes, state_for
from self_check import validate_frozen_state, validate_hif4_params

RECORDS = {}


def flat_codes(params):
    return (params["sign"].float() * params["mant"].float() * 4).flatten(-4, -1)


def proposals(module, params, values, state):
    decoded = module._linear_score_decode(params).float()
    deployed = state["unrestricted_deployed_weight"].float()
    target = values.float() @ state["v421_teacher_weight"].float().T
    gradient = (target - decoded @ deployed.T) @ deployed
    scales = [params[key].expand_as(params["mant"]).reshape_as(decoded)
              for key in ("scale_lv2", "scale_lv3", "scale_factor")]
    return module._v423_pair_code_proposals(flat_codes(params), decoded, gradient,
                                          state["linear_full_gram"], scales)


class V423Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.parent = load(422)
        cls.child = load(423)

    def test_parent_byte_prefix_and_import_scope(self):
        parent = (ROOT / "solution/solution_v422.py").read_bytes()
        child = (ROOT / "solution/solution_v423.py").read_bytes()
        self.assertTrue(child.startswith(parent))
        suffix = ast.parse(child[len(parent):].decode())
        functions = [node.name for node in suffix.body if isinstance(node, ast.FunctionDef)]
        self.assertEqual(functions, ["_v423_pair_code_proposals", "_v423_refine_linear_pairs",
                                     "hif4_dynamic_quantize_activation"])
        self.assertFalse(any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(suffix)))

    def test_225_code_cost_matches_explicit_bf16_decode_and_two_distinct_minima(self):
        weight = torch.zeros((2, 128))
        weight[:, 0], weight[:, 64] = torch.tensor([1., .5]), torch.tensor([.75, 1.])
        state = state_for(weight)
        values = torch.zeros((1, 128))
        values[0, 0], values[0, 64] = .375, -.6875
        original = torch.zeros((1, 128))
        original[0, 0], original[0, 64] = 3, -2
        params = params_from_codes(original)
        params["scale_factor"].fill_(1.25)
        params["scale_lv2"].fill_(2.)
        params["scale_lv3"].fill_(1.)
        self.assertEqual(validate_hif4_params(params, original.shape, "enumeration-parent"), [])
        got_codes, got_changes = proposals(self.child, params, values, state)
        legal = torch.arange(-7, 8).float()
        explicit_codes = original.repeat(225, 1)
        explicit_codes[:, 0] = legal[:, None].expand(15, 15).flatten()
        explicit_codes[:, 64] = legal[None].expand(15, 15).flatten()
        explicit_params = params_from_codes(explicit_codes)
        for key in ("scale_factor", "scale_lv2", "scale_lv3"):
            explicit_params[key] = params[key].repeat(225, *([1] * (params[key].ndim - 1)))
        self.assertEqual(validate_hif4_params(explicit_params, explicit_codes.shape, "enumeration-225"), [])
        deployed = state["unrestricted_deployed_weight"].float()
        target = values @ deployed.T
        before = self.child._linear_score_decode(params).float() @ deployed.T
        output = self.child._linear_score_decode(explicit_params).float() @ deployed.T
        explicit_changes = (target - output).square().sum(-1) - (target - before).square().sum(-1)
        top = explicit_changes.topk(2, largest=False).values
        torch.testing.assert_close(got_changes[:, 0], top, atol=2e-6, rtol=2e-5)
        self.assertFalse(torch.equal(got_codes[0], got_codes[1]))
        for index in range(2):
            match = (explicit_codes == got_codes[index]).all(-1)
            self.assertEqual(int(match.sum()), 1)
            torch.testing.assert_close(got_changes[index, 0], explicit_changes[match][0], atol=2e-6, rtol=2e-5)
        RECORDS["enumeration"] = {"legal_combinations": 225, "distinct_retained": 2,
                                  "metadata_validated": True,
                                  "best_changes": got_changes[:, 0].tolist()}
        # Separate arithmetic stress case: these deliberately illegal metadata
        # values test rounding order only, not legal HiF4 mechanism coverage.
        stress_params = copy.deepcopy(explicit_params)
        stress_params["scale_factor"].fill_(1.1015625)
        stress_params["scale_lv2"].fill_(1.5)
        stress_params["scale_lv3"].fill_(.75)
        self.assertTrue(validate_hif4_params(stress_params, explicit_codes.shape, "illegal-arithmetic-stress"))
        actual = self.child._linear_score_decode(stress_params).float()
        naive = explicit_codes * .25 * 1.5 * .75 * 1.1015625
        self.assertGreater(int((actual != naive).sum()), 0)
        RECORDS["arithmetic_stress_only"] = {"metadata_legal": False,
                                            "scope": "BF16 rounding-order stress; not legal HiF4 coverage",
                                            "bf16_decode_differs_from_naive_step": True}

    def test_non_greedy_trap_crosses_64_block(self):
        weight = torch.zeros((2, 128))
        weight[:, 0], weight[:, 64] = torch.tensor([1., 1.]), torch.tensor([1., 0.])
        state = state_for(weight)
        state["v421_teacher_weight"].zero_()
        state["v421_teacher_weight"][:, 0] = torch.tensor([-.0625, .1875]).bfloat16()
        values = torch.zeros((1, 128), dtype=torch.bfloat16)
        values[0, 0] = 1
        params = params_from_codes(torch.zeros((1, 128)))
        target = values.float() @ state["v421_teacher_weight"].float().T
        before = target.square().sum()
        for channel in (0, 64):
            output = torch.arange(-7, 8)[:, None] * .25 * weight[:, channel]
            self.assertTrue(bool(((target - output).square().sum(-1) >= before).all()))
        result = self.child._v423_refine_linear_pairs(params, values, state)
        codes = flat_codes(result)
        self.assertEqual((float(codes[0, 0]), float(codes[0, 64])), (1., -1.))
        output = (self.child._linear_score_decode(result) @ weight.bfloat16().T).float()
        after = (target - output).square().sum()
        self.assertLess(float(after), float(before))
        self.assertEqual(validate_hif4_params(result, values.shape, "trap"), [])
        RECORDS["pair_trap"] = {"single_coordinate_improvement": False,
                                 "cross_block_channels": [0, 64],
                                 "before_sse": float(before), "after_sse": float(after)}

    def test_full_code_jump_exceeds_two_and_single_side_zero_is_legal(self):
        weight = torch.zeros((2, 128))
        weight[0, 0], weight[1, 64] = 1., 1.
        state = state_for(weight)
        params = params_from_codes(torch.zeros((1, 128)))
        values = torch.zeros((1, 128), dtype=torch.bfloat16)
        values[0, 0] = 1.5
        result = self.child._v423_refine_linear_pairs(params, values, state)
        codes = flat_codes(result)
        self.assertEqual(float(codes[0, 0]), 6.)
        self.assertEqual(float(codes[0, 64]), 0.)
        for key in ("scale_factor", "scale_lv2", "scale_lv3"):
            self.assertIs(result[key], params[key])

    def test_bf16_gemm_tie_rejects_prediction_win(self):
        weight = torch.zeros((128, 128))
        weight[0, 0], weight[0, 64] = 256., 1.
        values = torch.zeros((1, 128), dtype=torch.bfloat16)
        values[0, 0], values[0, 64] = 1., .25
        codes = torch.zeros((1, 128))
        codes[0, 0] = 4
        params, state = params_from_codes(codes), state_for(weight)
        _, changes = proposals(self.child, params, values, state)
        self.assertTrue(bool((changes < 0).any()))
        result = self.child._v423_refine_linear_pairs(params, values, state)
        self.assertIs(result, params)
        RECORDS["actual_bf16_tie_rejected"] = True

    def test_129_rows_dynamic_and_full_shape_acceptance(self):
        rows, inputs = 129, 128
        values = torch.zeros((rows, inputs), dtype=torch.bfloat16)
        values[:, 0], values[:, 64] = 1.5, .5
        scale = torch.ones((rows, inputs // 16), dtype=torch.bfloat16)
        params = params_from_codes(torch.zeros((rows, inputs)))
        state = state_for(torch.eye(inputs))
        observed = []
        decoder = self.child._linear_score_decode
        def decode(candidate):
            observed.append(candidate["mant"].shape[0])
            return decoder(candidate)
        with patch.object(self.child, "_v423_parent_activation", return_value=params):
            with patch.object(self.child, "_linear_score_decode", side_effect=decode):
                result = self.child.hif4_dynamic_quantize_activation(values, scale, state)
        self.assertTrue(observed)
        self.assertEqual(set(observed), {rows})
        self.assertTrue(bool((flat_codes(result)[:, 0] == 6).all()))
        self.assertTrue(bool(((flat_codes(result) != flat_codes(params)).sum(-1) <= 2).all()))
        self.assertEqual(validate_hif4_params(result, values.shape, "129-rows"), [])
        RECORDS["expanded_rows"] = {"rows": rows, "accepted_rows": int((flat_codes(result) != 0).any(-1).sum()),
                                    "decode_row_shapes": observed}

    def test_zero_nonfinite_missing_and_budget_fallback(self):
        params = params_from_codes(torch.zeros((1, 64)))
        values = torch.zeros((1, 64), dtype=torch.bfloat16)
        state = state_for(torch.eye(64))
        for candidate_state in ({}, dict(state, direct_output=True), dict(state, linear_full_gram=None), state):
            self.assertIs(self.child._v423_refine_linear_pairs(params, values, candidate_state), params)
        bad = values.clone()
        bad[0, 0] = float("nan")
        self.assertIs(self.child._v423_refine_linear_pairs(params, bad, state), params)
        with patch.object(self.child, "_v423_pair_code_proposals") as proposal:
            self.assertIs(self.child._v423_refine_linear_pairs(params, values.expand(1025, -1), state), params)
            big = torch.zeros((1, 1), dtype=torch.bfloat16).expand(8192, 4096)
            big_state = dict(state, v421_teacher_weight=big, unrestricted_deployed_weight=big,
                             linear_full_gram=torch.zeros((1, 1)).expand(4096, 4096))
            self.assertIs(self.child._v423_refine_linear_pairs(params, torch.zeros((65, 1)).expand(65, 4096), big_state), params)
            proposal.assert_not_called()
        with patch.object(self.child, "_v423_pair_code_proposals", side_effect=RuntimeError("synthetic numeric failure")):
            self.assertIs(self.child._v423_refine_linear_pairs(params, values, state), params)

    def test_two_proposals_deduplicate_single_coordinate_edits(self):
        state = state_for(torch.eye(64))
        values = torch.zeros((1, 64))
        values[0, 0] = 1.5
        params = params_from_codes(torch.zeros_like(values))
        codes, changes = proposals(self.child, params, values, state)
        self.assertTrue(bool(torch.isfinite(changes).all()))
        self.assertFalse(torch.equal(codes[0], codes[1]))
        self.assertEqual(float(codes[0, 0, 0]), 6.)
        self.assertEqual(int((codes[0, 0] != 0).sum()), 1)

    def test_marker_absent_calibration_is_complete_parent_bitexact(self):
        generator = torch.Generator().manual_seed(423)
        def packed(rows):
            choices = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.bfloat16)
            quant = choices[torch.randint(0, 8, (rows, 64), generator=generator)]
            signs = 2 * torch.randint(0, 2, (rows, 64), generator=generator) - 1
            return (quant * signs.bfloat16()).contiguous(), torch.ones((rows, 4), dtype=torch.bfloat16)
        weight, calibration = packed(128), [packed(2), packed(2)]
        frozen = copy.deepcopy((weight, calibration))
        rng = torch.get_rng_state().clone()
        parent = self.parent.hif4_calibration_and_quantize_weight(*weight, calibration)
        parent_rng = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        with patch.object(self.child, "_v423_refine_linear_pairs") as refine:
            result = self.child.hif4_calibration_and_quantize_weight(*weight, calibration)
            refine.assert_not_called()
        self.assertTrue(torch.equal(parent_rng, torch.get_rng_state()))
        assert_tree_equal(self, result, parent)
        assert_tree_equal(self, (weight, calibration), frozen)
        self.assertEqual(validate_frozen_state(result["activation_state"], "v423-state"), [])
        activation = packed(129)
        frozen_dynamic = copy.deepcopy((activation, result["activation_state"]))
        rng = torch.get_rng_state().clone()
        dynamic = self.child.hif4_dynamic_quantize_activation(*activation, result["activation_state"])
        fresh = load(423, "fresh")
        replay = fresh.hif4_dynamic_quantize_activation(*activation, copy.deepcopy(result["activation_state"]))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        assert_tree_equal(self, dynamic, replay)
        assert_tree_equal(self, (activation, result["activation_state"]), frozen_dynamic)
        parent_dynamic = self.parent.hif4_dynamic_quantize_activation(*activation, parent["activation_state"])
        signed_changes = (flat_codes(dynamic) != flat_codes(parent_dynamic)).sum(-1)
        self.assertTrue(bool((signed_changes <= 2).all()))
        deployed = self.child._linear_score_decode(result["weight_params"])
        target = self.child.dequantize_nvfp4(*activation).float() @ self.child.dequantize_nvfp4(*weight).float().T
        losses = []
        for candidate in (parent_dynamic, dynamic):
            output = (self.child._linear_score_decode(candidate) @ deployed.T).float()
            losses.append((target - output).square().mean(-1))
        self.assertTrue(bool((losses[1] <= losses[0]).all()))
        self.assertEqual(validate_hif4_params(dynamic, activation[0].shape, "actual-dynamic"), [])
        RECORDS["synthetic_calibration"] = {"complete_parent_bitexact": True, "new_refine_calls_during_calibration": 0,
                                             "rows": 129, "improved_rows": int((losses[1] < losses[0]).sum()),
                                             "degraded_rows": int((losses[1] > losses[0]).sum()),
                                             "maximum_signed_code_changes_per_row": int(signed_changes.max()),
                                             "parent_mse": float(losses[0].mean()), "candidate_mse": float(losses[1].mean()),
                                             "fresh_module_equal": True, "input_state_rng_unchanged": True}


if __name__ == "__main__":
    outcome = unittest.main(verbosity=2, exit=False).result
    RECORDS["summary"] = {"tests": outcome.testsRun, "failures": len(outcome.failures),
                           "errors": len(outcome.errors), "passed": outcome.wasSuccessful(),
                           "scope": "synthetic mechanism and contracts; no official or local bank evaluation"}
    (ROOT / "docs/v423-tests.json").write_text(json.dumps(RECORDS, indent=2) + "\n")
    sys.exit(not outcome.wasSuccessful())
