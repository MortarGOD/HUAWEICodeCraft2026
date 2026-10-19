#!/usr/bin/env python3
"""Bounded synthetic v425 mechanisms; no official check or local model bank."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.test_v422 import load, assert_tree_equal
from tools.test_v421 import params_from_codes
from tools.test_v424 import packed_pair
from self_check import validate_frozen_state, validate_hif4_params

RECORDS = {}


def barrier_problem(bf16_tie=False):
    codes = torch.zeros((128, 64)); codes[0, :2] = 7.
    if bf16_tie:
        codes[0, 4] = 4.
    params = params_from_codes(codes)
    params['scale_factor'].fill_(4.)
    z = torch.zeros((4, 64))
    z[0, :2] = torch.tensor([1., -1.]); z[1, :2] = .25
    z[2, 2] = 1.; z[3, 3] = 1.
    if bf16_tie:
        z[:2, 4] = 512.
    teacher = codes.clone(); teacher[0, :2] = 11.
    target = z @ teacher.T
    samples = [(z.bfloat16(), target, torch.tensor(1.)) for _ in range(4)]
    return params, samples, teacher


class V425Mechanisms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.parent = load(422, '425parent')
        cls.child = load(425)

    def test_source_prefix_and_only_calibration_public_override(self):
        parent = (ROOT / 'solution/solution_v422.py').read_bytes()
        child = (ROOT / 'solution/solution_v425.py').read_bytes()
        self.assertTrue(child.startswith(parent))
        suffix = ast.parse(child[len(parent):].decode())
        self.assertFalse(any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(suffix)))
        functions = [n.name for n in suffix.body if isinstance(n, ast.FunctionDef)]
        self.assertEqual([n for n in functions if not n.startswith('_v425_')], ['hif4_calibration_and_quantize_weight'])
        self.assertFalse(any(isinstance(node, ast.Global) for node in ast.walk(suffix)))
        for name in ('hif4_dynamic_quantize_activation', 'hif4_calibration_attention',
                     'hif4_dynamic_quantize_q', 'hif4_dynamic_quantize_k', 'hif4_dynamic_quantize_v'):
            self.assertEqual(getattr(self.child, name).__code__.co_code, getattr(self.parent, name).__code__.co_code)
        RECORDS['scope'] = {'parent_byte_prefix': True, 'public_dynamic_and_attention_unchanged': True,
                            'new_imports': [], 'new_global_state': False}

    def test_actual_decode_and_legal_joint_saturated_barrier(self):
        params, samples, teacher = barrier_problem()
        before = self.child._linear_score_decode(params)
        z, target, _ = samples[0]
        original_sse = float(((z @ before.T).float() - target).square().sum())
        self.assertEqual(original_sse, 4.)
        individual = []
        for column in (0, 1):
            for new in (6., 8.):
                changed = before.clone(); changed[0, column] = new
                individual.append(float(((z @ changed.T).float() - target).square().sum()))
        self.assertTrue(all(value > original_sse for value in individual))
        frozen = copy.deepcopy((params, samples))
        rng = torch.get_rng_state().clone()
        result = self.child._v425_fixed_z_refine(params, samples)
        self.assertEqual(result['proposed_rows'], 1)
        self.assertEqual(result['accepted_rows'], 1)
        candidate = result['weight_params']
        decoded = self.child._linear_score_decode(candidate)
        self.assertEqual(decoded[0, :4].tolist(), [8., 8., 0., 0.])
        self.assertEqual(validate_hif4_params(candidate, before.shape, 'joint-candidate'), [])
        for name in ('scale_factor', 'scale_lv2'):
            self.assertTrue(torch.equal(candidate[name], params[name]))
        changes = (candidate['scale_lv3'] != params['scale_lv3']).flatten(1).sum(1)
        self.assertLessEqual(int(changes.max()), 1)
        self.assertEqual(float(candidate['scale_lv3'].flatten(1)[0, 0]), 2.)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        assert_tree_equal(self, (params, samples), frozen)
        RECORDS['saturated_joint_barrier'] = {
            'parent_sse': original_sse, 'single_coordinate_sse': individual,
            'joint_sse': float(((z @ decoded.T).float() - target).square().sum()),
            'proposed_rows': result['proposed_rows'], 'accepted_rows': result['accepted_rows'],
            'before_folds': result['before'][:, 0].tolist(), 'after_folds': result['after'][:, 0].tolist(),
            'sf_lv2_fixed': True, 'maximum_changed_microgroups_per_output': int(changes.max()),
            'input_state_rng_unchanged': True,
        }

    def test_real_bf16_gemm_tie_rejects_positive_float_proposal(self):
        params, samples, _ = barrier_problem(bf16_tie=True)
        parent = self.child._linear_score_decode(params)
        z, target, _ = samples[0]
        changed = parent.clone(); changed[0, :2] = 8.
        float_before = float((z.float() @ parent.float().T - target).square().sum())
        float_after = float((z.float() @ changed.float().T - target).square().sum())
        self.assertLess(float_after, float_before)
        self.assertTrue(torch.equal(z @ parent.T, z @ changed.T))
        result = self.child._v425_fixed_z_refine(params, samples)
        self.assertGreater(result['proposed_rows'], 0)
        self.assertEqual(result['accepted_rows'], 0)
        self.assertIsNone(result['weight_params'])
        RECORDS['bf16_tie_rejection'] = {'float_sse_before': float_before, 'float_sse_after': float_after,
                                        'actual_bf16_outputs_equal': True, 'proposed_rows': result['proposed_rows'],
                                        'accepted_rows': result['accepted_rows']}

    def test_two_fold_increment_full_residual_cross_terms_and_upper_bound(self):
        g = torch.Generator().manual_seed(42520)
        codes = torch.randint(-7, 8, (5, 128), generator=g).float()
        params = params_from_codes(codes)
        params['scale_factor'] = torch.tensor([1., 1.25, 2.5, 4., 8.]).reshape(5, 1, 1, 1, 1).expand(5, 2, 1, 1, 1).clone().bfloat16()
        params['scale_lv2'].copy_(torch.randint(1, 3, params['scale_lv2'].shape, generator=g))
        params['scale_lv3'].copy_(torch.randint(1, 3, params['scale_lv3'].shape, generator=g))
        b = self.child._linear_score_decode(params).float()
        gradients, hessians, targets, zs = [], [], [], []
        for fold in range(2):
            z = torch.randn((11 + fold, 128), generator=g).bfloat16().float()
            teacher = b + .5 * torch.randn(b.shape, generator=g)
            target = z @ teacher.T
            residual = target - z @ b.T
            gradients.append((residual.T @ z).reshape(5, 32, 4))
            zg = z.reshape(-1, 32, 4)
            hessians.append(torch.einsum('ngi,ngj->gij', zg, zg))
            targets.append(target); zs.append(z)
        grad, hessian = torch.stack(gradients), torch.stack(hessians)
        sf = params['scale_factor'].expand_as(params['mant']).reshape(5, 32, 4)
        lv2 = params['scale_lv2'].expand_as(params['mant']).reshape(5, 32, 4)
        lv3 = params['scale_lv3'].reshape(5, 32, 1)
        proposal = self.child._v425_microgroup_proposals(b.reshape(5, 32, 4), grad, hessian, sf, lv2, lv3)
        new_lv3 = 3. - lv3
        step = self.child._v425_decode_codes(torch.ones_like(sf), sf, lv2, new_lv3).float()
        ratio = b.reshape(5, 32, 4) / step
        low, high = ratio.floor().clamp(-7, 7), ratio.ceil().clamp(-7, 7)
        dl = self.child._v425_decode_codes(low, sf, lv2, new_lv3).float() - b.reshape(5, 32, 4)
        dh = self.child._v425_decode_codes(high, sf, lv2, new_lv3).float() - b.reshape(5, 32, 4)
        upper = torch.maximum(2 * grad * dl, 2 * grad * dh).sum(-1)
        max_error, min_upper_margin = 0., float('inf')
        cross_term_seen = False
        for row in range(5):
            group = int(proposal['groups'][row])
            for bits in range(16):
                select = torch.tensor([(bits >> bit) & 1 for bit in range(4)]).bool()
                code = torch.where(select, high[row, group], low[row, group])
                new = self.child._v425_decode_codes(code, sf[row, group], lv2[row, group], new_lv3[row, group]).float()
                delta = (new - b[row, 4*group:4*group+4]).double()
                for fold in range(2):
                    z = zs[fold].double(); target = targets[fold][:, row].double()
                    old_output = z @ b[row].double()
                    changed_output = old_output + z[:, 4*group:4*group+4] @ delta
                    direct = (target-old_output).square().sum() - (target-changed_output).square().sum()
                    gg = z[:, 4*group:4*group+4].T @ (target-old_output)
                    hh = z[:, 4*group:4*group+4].T @ z[:, 4*group:4*group+4]
                    formula = 2 * gg @ delta - delta @ hh @ delta
                    self.assertAlmostEqual(float(direct), float(formula), places=8)
                    float_formula = 2 * grad[fold, row, group].double() @ delta - delta @ hessian[fold, group].double() @ delta
                    error = float((formula-float_formula).abs())
                    self.assertLess(error, .005)
                    max_error = max(max_error, error)
                    margin = float(upper[fold, row, group]) - float(direct)
                    self.assertGreaterEqual(margin, -.005)
                    min_upper_margin = min(min_upper_margin, margin)
                    diagonal_only = 2 * gg @ delta - (hh.diagonal() * delta.square()).sum()
                    cross_term_seen |= abs(float(diagonal_only-formula)) > .01
        self.assertTrue(cross_term_seen)
        self.assertTrue(bool((proposal['shortlist_count'] <= 4).all()))
        RECORDS['quadratic_and_upper_bound'] = {'direct_float64_identity': True,
            'max_fp32_increment_error': max_error, 'minimum_upper_bound_margin': min_upper_margin,
            'nonzero_within_group_cross_terms': cross_term_seen,
            'shortlist_count': proposal['shortlist_count'].tolist()}

    def test_zero_scales_and_no_potential_do_not_propose(self):
        deployed = torch.zeros((3, 16, 4))
        gradients = torch.ones((2, 3, 16, 4))
        hessian = torch.eye(4).expand(2, 16, 4, 4)
        result = self.child._v425_microgroup_proposals(deployed, gradients, hessian,
                                                      torch.zeros_like(deployed), torch.ones_like(deployed), torch.ones((3,16,1)))
        self.assertFalse(bool(result['take'].any()))
        self.assertEqual(int(result['shortlist_count'].sum()), 0)
        params = params_from_codes(torch.zeros((128, 64)))
        samples = [(torch.zeros((4,64), dtype=torch.bfloat16), torch.zeros((4,128)), torch.tensor(1.))] * 4
        self.assertIsNone(self.child._v425_fixed_z_refine(params, samples))

    def test_shape_and_final_full_sequence_budgets_skip_extra_work(self):
        base_params = params_from_codes(torch.zeros((128, 64)))
        def fake_parent(inputs=64):
            return {'weight_params': base_params, 'activation_state': {
                'linear_shape': 1, 'linear_full_gram': torch.empty((inputs, inputs), device='meta'),
                'unrestricted_target_weight': torch.empty((1, inputs), device='meta')}}
        def sample(rows, inputs=64):
            return torch.empty((rows, inputs), device='meta'), torch.empty((rows, inputs//16), device='meta')
        cases = [(fake_parent(), torch.empty((128,64),device='meta'), [sample(8)] * 4),
                 (fake_parent(), torch.empty((128,64),device='meta'), [sample(8)] * 4 + [sample(1025)]),
                 (fake_parent(), torch.empty((128,64),device='meta'), [sample(8)] * 4 + [sample(1)]),
                 (fake_parent(4096), torch.empty((8192,4096),device='meta'), [sample(8,4096)] * 4 + [sample(65,4096)]),
                 (fake_parent(), torch.empty((8193,64),device='meta'), [sample(8)] * 5),
                 (fake_parent(), torch.empty((64,64),device='meta'), [sample(8)] * 5)]
        with patch.object(self.child, 'hif4_dynamic_quantize_activation', side_effect=AssertionError('budget guard must run first')):
            with patch.object(self.child, '_v425_fixed_z_refine', side_effect=AssertionError('budget guard must run first')):
                for parent, weight, calibration in cases:
                    self.assertIs(self.child._v425_calibration_probe(parent, weight, None, calibration), parent)
        RECORDS['budget_guards'] = {'cases': len(cases), 'extra_dynamic_and_search_calls': 0,
                                    'overbudget_final_sequence_not_truncated': True}

    def test_chunk_bounds_and_same_shape_fixed_z_gemms(self):
        params, samples, _ = barrier_problem()
        extended = {k: torch.cat([v, v, v[:1]], dim=0) for k,v in params.items()}
        samples = [(z, torch.cat([target, target, target[:,:1]], 1), norm) for z,target,norm in samples]
        with patch.object(self.child, '_v425_microgroup_proposals', wraps=self.child._v425_microgroup_proposals) as proposals:
            with patch.object(self.child, '_v425_fixed_z_losses', wraps=self.child._v425_fixed_z_losses) as losses:
                result = self.child._v425_fixed_z_refine(extended, samples)
        self.assertEqual([call.args[0].shape[0] for call in proposals.call_args_list], [128,128,1])
        self.assertEqual(losses.call_count, 2)
        self.assertEqual(result['accepted_rows'], 3)
        for call in losses.call_args_list:
            self.assertEqual(call.args[0]['mant'].shape[0], 257)
            self.assertEqual([s[0].shape[0] for s in call.args[1]], [4]*4)
        RECORDS['memory_and_gemm_shape'] = {'output_chunks': [128,128,1], 'full_bf16_output_width': 257,
                                          'sequence_rows_preserved': [4]*4}

    def test_full_real_calibration_search_recompile_dynamic_accept_and_reject(self):
        records = []
        for seed in (425, 4251, 4252, 4253, 4254, 4255):
            generator = torch.Generator().manual_seed(seed)
            weight = packed_pair(128, 64, generator)
            calibration = [packed_pair(8, 64, generator) for _ in range(5)]
            frozen_inputs = copy.deepcopy((weight, calibration))
            trace = {}
            parent_api = self.child._v425_parent_calibration
            refine_api = self.child._v425_fixed_z_refine
            guard_api = self.child._v425_last_calibration_guard
            def parent_call(*args):
                result = parent_api(*args)
                trace['parent'] = result
                trace['parent_frozen'] = copy.deepcopy(result)
                trace['rng'] = torch.get_rng_state().clone()
                return result
            def refine_call(*args):
                result = refine_api(*args); trace['refine'] = result; return result
            def guard_call(*args):
                result = guard_api(*args); trace['guard'] = result; trace['candidate'] = args[1]; return result
            with patch.object(self.child, '_v425_parent_calibration', side_effect=parent_call), \
                 patch.object(self.child, '_v425_fixed_z_refine', side_effect=refine_call), \
                 patch.object(self.child, '_v425_last_calibration_guard', side_effect=guard_call):
                result = self.child.hif4_calibration_and_quantize_weight(*weight, calibration)
            self.assertTrue(torch.equal(trace['rng'], torch.get_rng_state()))
            assert_tree_equal(self, trace['parent'], trace['parent_frozen'])
            assert_tree_equal(self, (weight, calibration), frozen_inputs)
            self.assertIn('guard', trace)
            candidate = trace['candidate']
            state = candidate['activation_state']; parent_state = trace['parent']['activation_state']
            self.assertEqual(validate_frozen_state(state, 'refreshed-state'), [])
            self.assertEqual(validate_hif4_params(candidate['weight_params'], weight[0].shape, 'refreshed-W'), [])
            b = self.child._linear_score_decode(candidate['weight_params']).float()
            self.assertTrue(torch.equal(state['unrestricted_deployed_weight'].float(), b))
            torch.testing.assert_close(state['linear_full_gram'], b.T @ b, atol=0, rtol=0)
            block_gram = torch.einsum('obi,obj->bij', b.reshape(128,1,64), b.reshape(128,1,64))
            torch.testing.assert_close(state['unrestricted_block_gram'], block_gram, atol=0, rtol=0)
            self.assertFalse(torch.equal(state['unrestricted_continuous_projection'], parent_state['unrestricted_continuous_projection']))
            for name in ('coefficient','permutation','hadamard','unrestricted_pair_transform',
                         'unrestricted_target_weight','v421_teacher_weight','v421_prefix_enabled'):
                if name in parent_state:
                    assert_tree_equal(self, state[name], parent_state[name])
            for name in ('scale_factor','scale_lv2'):
                self.assertTrue(torch.equal(candidate['weight_params'][name], trace['parent']['weight_params'][name]))
            micro_changes = (candidate['weight_params']['scale_lv3'] != trace['parent']['weight_params']['scale_lv3']).flatten(1).sum(1)
            self.assertLessEqual(int(micro_changes.max()), 1)
            self.assertIs(result, candidate if trace['guard']['accept'] else trace['parent'])
            last = calibration[-1]
            dynamic = self.child.hif4_dynamic_quantize_activation(*last, result['activation_state'])
            replay = load(425, f'fresh_{seed}').hif4_dynamic_quantize_activation(*last, copy.deepcopy(result['activation_state']))
            assert_tree_equal(self, dynamic, replay)
            self.assertEqual(validate_hif4_params(dynamic, last[0].shape, 'dynamic'), [])
            records.append({'seed':seed, 'fp32_proposed_rows':trace['refine']['proposed_rows'],
                'fixed_z_bf16_accepted_rows':trace['refine']['accepted_rows'],
                'fixed_z_before_fold_sum':trace['refine']['before'].sum(1).tolist(),
                'fixed_z_after_fold_sum':trace['refine']['after'].sum(1).tolist(),
                'reencoded_guard_before':trace['guard']['before'].tolist(),
                'reencoded_guard_after':trace['guard']['after'].tolist(),
                'accepted':trace['guard']['accept'], 'maximum_changed_microgroups_per_output':int(micro_changes.max())})
        self.assertEqual(sum(record['accepted'] for record in records), 1)
        self.assertEqual(records[-1]['seed'], 4255)
        self.assertTrue(records[-1]['accepted'])
        RECORDS['real_synthetic_pipeline'] = {'cases':records, 'real_search_installer_public_dynamic_guard':True,
            'no_algorithm_results_mocked':True, 'refreshed_state_valid':True,
            'parent_input_state_and_extra_rng_unchanged':True, 'fresh_module_replay_equal':True,
            'scope':'Six small synthetic draws retained, including five rejects; selected to cover both branches, not evidence of unseen quality.'}

    def test_down_exact_exception_parent_fallback_and_state_failure(self):
        g = torch.Generator().manual_seed(42530)
        weight = packed_pair(64, 128, g); calibration = [packed_pair(4,128,g) for _ in range(5)]
        parent = self.parent.hif4_calibration_and_quantize_weight(*weight, calibration)
        with patch.object(self.child, '_v425_fixed_z_refine', side_effect=AssertionError('down should skip')):
            candidate = self.child.hif4_calibration_and_quantize_weight(*weight, calibration)
        assert_tree_equal(self, candidate, parent)
        a = packed_pair(3,128,g)
        assert_tree_equal(self, self.child.hif4_dynamic_quantize_activation(*a,candidate['activation_state']),
                          self.parent.hif4_dynamic_quantize_activation(*a,parent['activation_state']))
        sentinel = object()
        for error in (RuntimeError('numeric failure'), ValueError('invalid finite data')):
            with patch.object(self.child, '_v425_parent_calibration', return_value=sentinel), \
                 patch.object(self.child, '_v425_calibration_probe', side_effect=error):
                self.assertIs(self.child.hif4_calibration_and_quantize_weight(None,None,None), sentinel)
        params, _, teacher = barrier_problem()
        parent_state = {'unrestricted_target_weight':teacher.bfloat16(),
                        'unrestricted_continuous_projection':torch.full((64,64),float('nan')),
                        'linear_full_gram':torch.full((64,64),float('nan')),
                        'unrestricted_super_gram':torch.ones((1,128,128)),
                        'unrestricted_super_inverse':torch.ones((1,128,128))}
        # All-zero microgroups make the legacy compiler's inverse nonfinite;
        # v425 correctly refuses that rebuilt state. Give every group support
        # before testing stale-field replacement itself.
        self.assertIsNone(self.child._v425_recompile_state(parent_state, params))
        code = self.child._linear_score_decode(params).float()
        code[torch.arange(1,65), torch.arange(64)] = 1.
        params = params_from_codes(code); params['scale_factor'].fill_(4.)
        teacher[torch.arange(1,65), torch.arange(64)] = 1.
        parent_state['unrestricted_target_weight'] = teacher.bfloat16()
        old = copy.deepcopy(parent_state)
        rebuilt = self.child._v425_recompile_state(parent_state, params)
        self.assertIsNotNone(rebuilt)
        self.assertTrue(bool(torch.isfinite(rebuilt['unrestricted_continuous_projection']).all()))
        self.assertNotIn('unrestricted_super_gram', rebuilt)
        self.assertNotIn('unrestricted_super_inverse', rebuilt)
        self.assertTrue(torch.isnan(parent_state['linear_full_gram']).all())
        self.assertTrue(torch.isnan(old['linear_full_gram']).all())
        RECORDS['fallback_and_stale_state'] = {'down_parent_exact':True, 'exception_parent_exact':True,
            'stale_projection_and_full_gram_replaced':True, 'stale_super_state_removed':True,
            'nonfinite_recompiled_micro_inverse_rejected':True}


if __name__ == '__main__':
    result = unittest.main(verbosity=2, exit=False).result
    RECORDS['summary'] = {'tests':result.testsRun, 'failures':len(result.failures), 'errors':len(result.errors),
        'passed':result.wasSuccessful(), 'source_sha256':hashlib.sha256((ROOT/'solution/solution_v425.py').read_bytes()).hexdigest(),
        'scope':'Synthetic mechanisms, legal parameters, complete small API paths; no official self-check or local model bank.'}
    (ROOT/'docs/v425-mechanisms.json').write_text(json.dumps(RECORDS,indent=2)+'\n')
    sys.exit(not result.wasSuccessful())
