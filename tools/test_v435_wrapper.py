"""Bounded v435 wrapper checks with a mixed real HiF4 head layout.

The parent dynamic is stubbed to isolate the wrapper; the deployed refinement,
NVFP4 decoder and public HiF4 decoder are real. No dataset or calibration run.
"""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hif4_quant import dequantize_hif4, dequantize_nvfp4
from self_check import validate_hif4_params
from tools.test_v422 import assert_tree_equal, load


def mixed_fixture():
    rows, heads, dim = 4, 2, 128
    x = torch.zeros(rows, heads * dim)
    for offset in (0, dim):
        x[:, offset] = .125
        x[:, offset + 1] = torch.tensor([1., 1., -1., -1.])
        x[:, offset + 2] = torch.tensor([.75, .75, -.75, -.75])
    # Legal NVFP4 levels: .125=.5*.25, 1=4*.25 and .75=3*.25.
    quant = (x * 4).bfloat16()
    scale = torch.full((rows, heads * dim // 16), .25, dtype=torch.bfloat16)
    values = x.clone()
    for offset in (0, dim):
        values[:, offset] = torch.tensor([0., 0., .25, .25])
    blocks = heads * dim // 64
    params = {
        'scale_factor': torch.ones(rows, blocks, 1, 1, 1, dtype=torch.bfloat16),
        'scale_lv2': torch.ones(rows, blocks, 8, 1, 1, dtype=torch.bfloat16),
        'scale_lv3': torch.ones(rows, blocks, 8, 2, 1, dtype=torch.bfloat16),
        'sign': values.sign().reshape(rows, blocks, 8, 2, 4).bfloat16(),
        'mant': values.abs().reshape(rows, blocks, 8, 2, 4).bfloat16(),
    }
    original = {'kernel': torch.zeros(2, 1, 1),
                'active': torch.tensor([True, False])}
    return quant, scale, params, original


class V435Wrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def setUp(self):
        self.module = load(435, self._testMethodName)

    def test_mixed_kernel_heads_use_real_refinement_on_inactive_head(self):
        module = self.module
        quant, scale, params, original = mixed_fixture()
        state = {'v435_content_parent': original}
        frozen = copy.deepcopy((quant, scale, params, state))
        rng = torch.get_rng_state().clone()
        with patch.object(module, '_v435_parent_dynamic', return_value=params) as parent:
            with patch.object(module, '_v435_refine', wraps=module._v435_refine) as refine:
                result = module.hif4_dynamic_quantize_v(quant, scale, 2, 128, state)
        self.assertIs(parent.call_args.args[-1], original)
        self.assertTrue(torch.equal(refine.call_args.args[2], torch.tensor([False, True])))
        self.assertEqual(validate_hif4_params(result, quant.shape, 'mixed-wrapper'), [])
        before = dequantize_hif4(params, quant.shape).double()
        after = dequantize_hif4(result, quant.shape).double()
        self.assertTrue(torch.equal(before[:, :128], after[:, :128]))
        self.assertFalse(torch.equal(before[:, 128:], after[:, 128:]))
        for name in params:
            self.assertTrue(torch.equal(params[name][:, :2], result[name][:, :2]))
        raw = dequantize_nvfp4(quant, scale).double()
        old_error, new_error = before - raw, after - raw
        self.assertTrue(bool((new_error.square().sum(0) <= old_error.square().sum(0)).all()))
        self.assertTrue(bool((new_error.sum(0).square() <= old_error.sum(0).square()).all()))
        assert_tree_equal(self, (quant, scale, params, state), frozen)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_parent_calibration_internal_unmarked_call_is_direct_passthrough(self):
        module = self.module
        quant, scale, params, original = mixed_fixture()
        parent_result = {'q_state': {'sentinel': 1}, 'k_state': {'sentinel': 2},
                         'v_state': original}
        frozen = copy.deepcopy(parent_result)
        sentinel_samples = []

        def parent_calibration(samples, qh, kh, dim):
            self.assertIs(samples, sentinel_samples)
            self.assertEqual((qh, kh, dim), (4, 2, 128))
            internal = module.hif4_dynamic_quantize_v(quant, scale, kh, dim, original)
            self.assertIs(internal, params)
            return parent_result

        with patch.object(module, '_v435_parent_dynamic', return_value=params) as dynamic:
            with patch.object(module, '_v435_parent_calibration', side_effect=parent_calibration):
                with patch.object(module, '_v435_refine', side_effect=AssertionError('internal activation')):
                    result = module.hif4_calibration_attention(sentinel_samples, 4, 2, 128)
        self.assertIs(dynamic.call_args.args[-1], original)
        self.assertIs(result['v_state']['v435_content_parent'], original)
        self.assertIs(result['q_state'], parent_result['q_state'])
        self.assertIs(result['k_state'], parent_result['k_state'])
        assert_tree_equal(self, parent_result, frozen)

    def test_all_kernel_heads_active_bypass(self):
        module = self.module
        quant, scale, params, original = mixed_fixture()
        original['active'] = torch.ones(2, dtype=torch.bool)
        frozen = copy.deepcopy(original)
        with patch.object(module, '_v435_parent_dynamic', return_value=params):
            with patch.object(module, '_v435_refine', side_effect=AssertionError('active VK changed')):
                result = module.hif4_dynamic_quantize_v(
                    quant, scale, 2, 128, {'v435_content_parent': original})
        self.assertIs(result, params)
        assert_tree_equal(self, original, frozen)

    def test_proposal_decode_matches_public_for_all_legal_scale_codes(self):
        # 255 SF values x 15 signed codes x four micro-scale combinations.
        module = self.module
        sf = module._decode_e6m2(torch.arange(255)).reshape(255, 1, 1)
        codes = torch.arange(-7, 8).reshape(1, 15, 1)
        lv2 = torch.tensor([1., 1., 2., 2.]).reshape(1, 1, 4)
        lv3 = torch.tensor([1., 2., 1., 2.]).reshape(1, 1, 4)
        for dtype in (torch.float32, torch.bfloat16):
            params = {'sign': codes.sign().to(dtype), 'mant': codes.abs().to(dtype) * .25,
                      'scale_factor': sf.to(dtype), 'scale_lv2': lv2.to(dtype),
                      'scale_lv3': lv3.to(dtype)}
            public = dequantize_hif4(params, (255, 60)).double()
            proposal = module._v435_decode_codes(
                codes, sf.double() * lv2.double() * lv3.double() * .25).reshape(255, 60)
            self.assertTrue(torch.equal(public, proposal))


if __name__ == '__main__':
    unittest.main(verbosity=2)
