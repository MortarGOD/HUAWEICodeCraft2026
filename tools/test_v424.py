#!/usr/bin/env python3
"""Synthetic v424 mechanism and contract checks; no official/local bank runs."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.test_v422 import assert_tree_equal, load
from self_check import validate_frozen_state, validate_hif4_params

RECORDS = {}


def natural_pair(rows=128, outputs=192, width=128, seed=424):
    rng = torch.Generator().manual_seed(seed)
    return torch.randn((rows, width), generator=rng), torch.randn((outputs, width), generator=rng)


def packed_pair(rows, width, generator):
    levels = torch.tensor([-6., -4., -3., -2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2., 3., 4., 6.]).bfloat16()
    quant = levels[torch.randint(len(levels), (rows, width), generator=generator)]
    return quant, torch.ones((rows, width // 16), dtype=torch.bfloat16)


class V424Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.parent = load(422)
        cls.child = load(424)

    def test_parent_byte_prefix_and_private_hook_scope(self):
        parent = (ROOT / 'solution/solution_v422.py').read_bytes()
        child = (ROOT / 'solution/solution_v424.py').read_bytes()
        self.assertTrue(child.startswith(parent))
        suffix = ast.parse(child[len(parent):].decode())
        self.assertFalse(any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(suffix)))
        names = [node.name for node in suffix.body if isinstance(node, ast.FunctionDef)]
        self.assertEqual(names, ['_v424_proxy_pair', '_v424_procrustes_step',
                                '_v424_two_fold_improves', '_v424_learn_block_transform',
                                '_v422_joint_block_transform'])
        public = ['hif4_calibration_and_quantize_weight', 'hif4_dynamic_quantize_activation',
                  'hif4_calibration_attention', 'hif4_dynamic_quantize_query',
                  'hif4_dynamic_quantize_key', 'hif4_dynamic_quantize_value']
        for name in public:
            if hasattr(self.parent, name):
                self.assertEqual(getattr(self.parent, name).__code__.co_code,
                                 getattr(self.child, name).__code__.co_code)
        RECORDS['scope'] = {'v422_byte_prefix': True, 'new_imports': [],
                            'public_entrypoints_unchanged': True}

    def test_procrustes_increment_minimizes_fixed_legal_codes(self):
        a, w = natural_pair(rows=48, outputs=96, seed=4241)
        qa_params = self.child._quantize_linear_values(a, offsets=(0,))
        qw_params = self.child._quantize_linear_values(w, offsets=(0,))
        for values, params in ((a, qa_params), (w, qw_params)):
            self.assertEqual(validate_hif4_params(params, values.shape, 'fixed-code'), [])
        qa, qw = map(self.child._linear_score_decode, (qa_params, qw_params))
        increment = self.child._v424_procrustes_step(a, w, qa, qw)
        eye = torch.eye(64).expand(2, 64, 64)
        torch.testing.assert_close(increment.transpose(-1, -2) @ increment, eye, atol=2e-6, rtol=2e-6)
        a_g, w_g = a[::2].reshape(-1, 2, 64), w.reshape(-1, 2, 64)
        qa_g, qw_g = qa[::2].float().reshape_as(a_g), qw.float().reshape_as(w_g)
        def objective(r):
            ar = torch.einsum('ngi,gij->ngj', a_g, r)
            wr = torch.einsum('ngi,gij->ngj', w_g, r)
            return (((ar - qa_g).square().sum((0, 2)) / a_g.square().sum((0, 2)))
                    + ((wr - qw_g).square().sum((0, 2)) / w_g.square().sum((0, 2))))
        before, after = objective(eye), objective(increment)
        self.assertTrue(bool((after <= before + 1e-7).all()))
        self.assertTrue(bool((after < before).all()))
        RECORDS['fixed_code_frobenius'] = {'before_per_block': before.tolist(),
                                          'after_per_block': after.tolist(),
                                          'orthogonality_max_error': float((increment.transpose(-1,-2) @ increment-eye).abs().max())}

    def test_frozen_right_composition_uses_inverse_pair_not_orthogonal_shortcut(self):
        a, w = natural_pair(rows=16, outputs=32, seed=4242)
        g = torch.Generator().manual_seed(42420)
        current = torch.linalg.qr(torch.randn((2, 64, 64), generator=g)).Q.bfloat16().float()
        step = torch.linalg.qr(torch.randn((2, 64, 64), generator=g)).Q
        frozen = (current @ step).bfloat16().float()
        result = self.child._v424_proxy_pair(a, w, frozen, a @ w.T)
        self.assertIsNotNone(result)
        transformed_a, transformed_w = result[:2]
        target = a @ w.T
        torch.testing.assert_close(transformed_a @ transformed_w.T, target, atol=2e-4, rtol=2e-4)
        wrong_w = torch.einsum('ogi,gij->ogj', w.reshape(-1, 2, 64), frozen).reshape_as(w)
        correct_mse = float((transformed_a @ transformed_w.T-target).square().mean())
        shortcut_mse = float((transformed_a @ wrong_w.T-target).square().mean())
        self.assertGreater(shortcut_mse, correct_mse * 1000)
        self.assertEqual(result[2].dtype, torch.bfloat16)
        self.assertEqual(result[3].dtype, torch.bfloat16)
        RECORDS['paired_inverse'] = {'continuous_mse': correct_mse,
                                     'incorrect_orthogonal_shortcut_mse': shortcut_mse,
                                     'frozen_matrix_is_actual_bf16': bool(torch.equal(frozen, frozen.bfloat16().float()))}

    def test_two_fold_gate_requires_strict_finite_improvement_in_both_folds(self):
        gate = self.child._v424_two_fold_improves
        parent = torch.tensor([2., 2.])
        self.assertTrue(gate(torch.tensor([1., 1.9]), parent))
        for candidate in ([1., 2.], [1., 3.], [2., 2.], [float('nan'), 1.], [1., float('inf')]):
            self.assertFalse(gate(torch.tensor(candidate), parent))

    def test_global_output_keeps_cross_block_cancellation(self):
        # Both arms have exactly representable legal HiF4 values. The helper
        # receives a fixed zero teacher here solely to isolate error reduction.
        a = torch.zeros((4, 128)); a[:, [0, 64]] = 1.
        small = torch.zeros((8, 128)); small[:, [0, 64]] = .25
        cancelling = torch.zeros((8, 128)); cancelling[:, 0] = 1.; cancelling[:, 64] = -1.
        identity = torch.eye(64).expand(2, 64, 64)
        target = torch.zeros((4, 8))
        before = self.child._v424_proxy_pair(a, small, identity, target)
        after = self.child._v424_proxy_pair(a, cancelling, identity, target)
        self.assertTrue(self.child._v424_two_fold_improves(after[-1], before[-1]))
        def independent_error(result):
            qa, qw = result[2:4]
            return sum(float((qa[:, b:b+64] @ qw[:, b:b+64].T).float().square().mean())
                       for b in (0, 64))
        independent_before, independent_after = map(independent_error, (before, after))
        self.assertGreater(independent_after, independent_before)
        self.assertEqual(after[-1].tolist(), [0., 0.])
        for values in (a, small, cancelling):
            self.assertEqual(validate_hif4_params(self.child._quantize_linear_values(values, offsets=(0,)), values.shape, 'cancellation'), [])
        RECORDS['cross_block_cancellation'] = {'whole_output_parent_folds': before[-1].tolist(),
                                                'whole_output_candidate_folds': after[-1].tolist(),
                                                'sum_independent_block_errors': [independent_before, independent_after]}

    def test_natural_isotropic_pair_learns_and_changes_real_legal_codes(self):
        a, w = natural_pair()
        parent = self.child._v424_parent_joint_block_transform(a, w)
        self.assertTrue(torch.equal(parent, torch.eye(64).expand(2, 64, 64)))
        initial = self.child._v424_proxy_pair(a, w, parent, a @ w.T)
        original_gate = self.child._v424_two_fold_improves
        gate_trace = []
        def tracked_gate(losses, previous):
            accepted = original_gate(losses, previous)
            gate_trace.append({'before': previous.tolist(), 'candidate': losses.tolist(), 'accepted': accepted})
            return accepted
        with patch.object(self.child, '_v424_proxy_pair', wraps=self.child._v424_proxy_pair) as proxy, \
             patch.object(self.child, '_v424_procrustes_step', wraps=self.child._v424_procrustes_step) as steps, \
             patch.object(self.child, '_v424_two_fold_improves', side_effect=tracked_gate):
            start = time.perf_counter()
            learned = self.child._v424_learn_block_transform(a, w, parent)
            elapsed = time.perf_counter() - start
        final = self.child._v424_proxy_pair(a, w, learned, a @ w.T)
        self.assertGreater(int((learned != parent).sum()), 0)
        self.assertTrue(self.child._v424_two_fold_improves(final[-1], initial[-1]))
        self.assertLessEqual(proxy.call_count, 5)
        self.assertLessEqual(steps.call_count, 2)
        changes = []
        for before, after in zip(initial[:2], final[:2]):
            old = self.child._quantize_linear_values(before, offsets=(0,))
            new = self.child._quantize_linear_values(after, offsets=(0,))
            self.assertEqual(validate_hif4_params(new, after.shape, 'learned'), [])
            changes.append(sum(int((old[k] != new[k]).sum()) for k in ('sign', 'mant')))
        self.assertTrue(all(changes))
        RECORDS['natural_isotropic'] = {'seed': 424, 'parent_spectral_map_identity': True,
                                        'parent_aw_folds': initial[-1].tolist(), 'learned_aw_folds': final[-1].tolist(),
                                        'learned_blocks': int((learned != parent).flatten(1).any(1).sum()),
                                        'changed_matrix_entries': int((learned != parent).sum()),
                                        'changed_sign_mantissa_entries_a_w': changes,
                                        'changed_decoded_entries_a_w': [int((initial[i] != final[i]).sum()) for i in (2, 3)],
                                        'proxy_calls': proxy.call_count, 'step_calls': steps.call_count,
                                        'accepted_proposals': sum(item['accepted'] for item in gate_trace),
                                        'rejected_proposals': sum(not item['accepted'] for item in gate_trace),
                                        'proposal_trace': gate_trace,
                                        'learn_seconds': elapsed}

    def test_zero_rank_deficient_nonfinite_and_shape_guards(self):
        identity = torch.eye(64)[None]
        z_a, z_w = torch.zeros((8, 64)), torch.zeros((16, 64))
        self.assertIs(self.child._v424_learn_block_transform(z_a, z_w, identity), identity)
        self.assertTrue(torch.equal(self.child._v424_procrustes_step(z_a, z_w, z_a.bfloat16(), z_w.bfloat16()), identity))
        rank_a, rank_w = z_a.clone(), z_w.clone()
        rank_a[:, 0] = torch.arange(8).float(); rank_w[:, 0] = torch.arange(16).float()
        rank_result = self.child._v424_learn_block_transform(rank_a, rank_w, identity)
        self.assertTrue(bool(torch.isfinite(rank_result).all()))
        wide_a, wide_w = torch.ones((4, 4160)), torch.ones((8, 4160))
        wide_f = torch.eye(64).expand(65, 64, 64)
        cases = [(z_a[:3], z_w, identity), (wide_a, wide_w, wide_f),
                 (torch.full_like(z_a, float('nan')), z_w, identity),
                 (z_a, torch.full_like(z_w, float('inf')), identity),
                 (z_a, z_w[:0], identity), (z_a[:, :32], z_w[:, :32], identity),
                 (z_a, z_w, None)]
        with patch.object(self.child, '_v424_proxy_pair') as proxy:
            for a, w, parent in cases:
                self.assertIs(self.child._v424_learn_block_transform(a, w, parent), parent)
            proxy.assert_not_called()
        RECORDS['degenerate_guards'] = {'zero_parent_exact': True, 'rank_deficient_finite': True,
                                       'unsupported_or_nonfinite_proxy_calls': 0}

    def test_null_rejected_and_solver_exception_keep_parent_exact(self):
        a, w = natural_pair(rows=8, outputs=16, width=64, seed=4243)
        parent = torch.eye(64)[None]
        for mock in (dict(return_value=None), dict(side_effect=RuntimeError('synthetic solver failure'))):
            with patch.object(self.child, '_v424_proxy_pair', **mock):
                self.assertIs(self.child._v424_learn_block_transform(a, w, parent), parent)
        with patch.object(self.child, '_v424_two_fold_improves', return_value=False):
            self.assertIs(self.child._v424_learn_block_transform(a, w, parent), parent)
        singular = torch.zeros_like(parent)
        self.assertIs(self.child._v424_learn_block_transform(a, w, singular), singular)
        with patch.object(self.child, '_v424_procrustes_step', side_effect=RuntimeError('synthetic SVD failure')):
            self.assertIs(self.child._v424_learn_block_transform(a, w, parent), parent)

    def test_inputs_rng_repeat_and_deterministic_sampling_bound(self):
        a, w = natural_pair(rows=137, outputs=271, width=64, seed=4244)
        frozen = copy.deepcopy((a, w))
        rng = torch.get_rng_state().clone()
        with patch.object(self.child, '_v424_proxy_pair', wraps=self.child._v424_proxy_pair) as proxy:
            first = self.child._v422_joint_block_transform(a, w)
        for args in proxy.call_args_list:
            self.assertEqual(args.args[0].shape, (128, 64))
            self.assertEqual(args.args[1].shape, (256, 64))
        fresh = load(424, 'natural_fresh')
        second = fresh._v422_joint_block_transform(a, w)
        self.assertTrue(torch.equal(first, second))
        assert_tree_equal(self, (a, w), frozen)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        RECORDS['sampling_and_purity'] = {'rows': 137, 'outputs': 271, 'sample_rows': 128,
                                          'sample_outputs': 256, 'fresh_module_exact': True,
                                          'inputs_and_rng_unchanged': True}

    def test_down_calibration_and_dynamic_remain_parent_exact(self):
        g = torch.Generator().manual_seed(4245)
        w, calibration = packed_pair(64, 128, g), [packed_pair(8, 128, g)]
        with patch.object(self.child, '_v424_learn_block_transform', side_effect=AssertionError('down must bypass learner')):
            result = self.child.hif4_calibration_and_quantize_weight(*w, calibration)
        expected = self.parent.hif4_calibration_and_quantize_weight(*w, calibration)
        assert_tree_equal(self, result, expected)
        a = packed_pair(5, 128, g)
        assert_tree_equal(self, self.child.hif4_dynamic_quantize_activation(*a, result['activation_state']),
                          self.parent.hif4_dynamic_quantize_activation(*a, expected['activation_state']))
        RECORDS['down'] = {'calibration_parent_exact': True, 'dynamic_parent_exact': True,
                           'learner_calls': 0}

    def test_small_actual_up_calibration_frozen_state_and_fresh_dynamic(self):
        g = torch.Generator().manual_seed(4246)
        w, calibration = packed_pair(128, 64, g), [packed_pair(8, 64, g), packed_pair(8, 64, g)]
        frozen = copy.deepcopy((w, calibration))
        rng = torch.get_rng_state().clone()
        with patch.object(self.child, '_v424_learn_block_transform', wraps=self.child._v424_learn_block_transform) as learner:
            result = self.child.hif4_calibration_and_quantize_weight(*w, calibration)
        self.assertEqual(learner.call_count, 1)
        child_calibration_rng = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        self.parent.hif4_calibration_and_quantize_weight(*w, calibration)
        self.assertTrue(torch.equal(child_calibration_rng, torch.get_rng_state()))
        # Legacy calibration reseeds the global generator. v424 adds no RNG
        # operation; compare this inherited effect and test dynamic purity below.
        rng = torch.get_rng_state().clone()
        state = result['activation_state']
        self.assertEqual(validate_frozen_state(state, 'v424-up-state'), [])
        self.assertEqual(validate_hif4_params(result['weight_params'], w[0].shape, 'v424-up-weight'), [])
        self.assertEqual(tuple(state['unrestricted_pair_transform'].shape), (1, 64, 64))
        self.assertEqual(state['unrestricted_pair_transform'].dtype, torch.bfloat16)
        assert_tree_equal(self, (w, calibration), frozen)
        a = packed_pair(3, 64, g)
        before_dynamic = copy.deepcopy((a, state))
        dynamic = self.child.hif4_dynamic_quantize_activation(*a, state)
        replay = load(424, 'api_fresh').hif4_dynamic_quantize_activation(*a, copy.deepcopy(state))
        assert_tree_equal(self, dynamic, replay)
        assert_tree_equal(self, (a, state), before_dynamic)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(validate_hif4_params(dynamic, a[0].shape, 'v424-up-dynamic'), [])
        RECORDS['synthetic_up_contract'] = {'learner_calls': learner.call_count,
                                             'weight_and_activation_legal': True,
                                             'frozen_state_valid': True, 'fresh_dynamic_exact': True,
                                             'input_state_unchanged': True, 'dynamic_rng_unchanged': True,
                                             'inherited_calibration_rng_effect_matches_parent': True}


if __name__ == '__main__':
    result = unittest.main(verbosity=2, exit=False).result
    RECORDS['summary'] = {'tests': result.testsRun, 'failures': len(result.failures),
                           'errors': len(result.errors), 'passed': result.wasSuccessful(),
                           'scope': 'synthetic mechanisms and contracts; no official or local bank evaluation',
                           'source_sha256': hashlib.sha256((ROOT / 'solution/solution_v424.py').read_bytes()).hexdigest()}
    (ROOT / 'docs/v424-tests.json').write_text(json.dumps(RECORDS, indent=2) + '\n')
    sys.exit(not result.wasSuccessful())
