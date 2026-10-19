"""Small independent checks of self-only content balancing; no bank evaluation."""

import copy
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.attention_content_balance_core import _v435_refine, _v435_buckets
from tools.test_v422 import load, assert_tree_equal
from self_check import validate_hif4_params


def decode(params, shape):
    # Public parameter arithmetic, independent of the core decoder.
    return (params['sign'] * params['mant'] * params['scale_lv2']
            * params['scale_lv3'] * params['scale_factor']).reshape(shape).to(torch.bfloat16).double()


def objective(error, buckets):
    # Explicit masks, no shared scatter/statistics implementation.
    total = error.sum(0)
    sse = error.square().sum(0)
    value = (total.square() + sse) / error.shape[0]
    for group in range(4):
        chosen = error[buckets == group]
        if len(chosen):
            value = value + chosen.sum(0).square() / len(chosen)
    return value, sse, total.square()


def fixture():
    x = torch.zeros(4, 128)
    x[:, 0] = 0.125
    x[:, 1] = torch.tensor([1., 1., -1., -1.])
    x[:, 2] = torch.tensor([0.75, 0.75, -0.75, -0.75])
    # The second head repeats the active example but must remain untouched.
    x[:, 64:] = x[:, :64]
    values = x.clone()
    values[:, 0] = torch.tensor([0., 0., 0.25, 0.25])
    values[:, 64] = values[:, 0]
    p = {'scale_factor': torch.ones(4, 2, 1, 1, 1, dtype=torch.bfloat16),
         'scale_lv2': torch.ones(4, 2, 8, 1, 1, dtype=torch.bfloat16),
         'scale_lv3': torch.ones(4, 2, 8, 2, 1, dtype=torch.bfloat16),
         'sign': values.sign().reshape(4, 2, 8, 2, 4).to(torch.bfloat16),
         'mant': values.abs().reshape(4, 2, 8, 2, 4).to(torch.bfloat16)}
    return x, p


class ContentBalance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_toy_content_gain_at_identical_global_mean_and_sse(self):
        # Two legal code arrangements prove headroom invisible to global mean
        # and SSE; one signed-code exchange reaches that same-cost solution.
        x, p = fixture()
        old = decode(p, x.shape)
        alternate = old.clone()
        alternate[:, 0] = torch.tensor([0., 0.25, 0., 0.25])
        bucket = _v435_buckets(x[:, :64])
        a = objective(old[:, :64] - x[:, :64], bucket)
        b = objective(alternate[:, :64] - x[:, :64], bucket)
        self.assertTrue(torch.equal(a[1], b[1]))
        self.assertTrue(torch.equal(a[2], b[2]))
        self.assertLess(float(b[0][0]), float(a[0][0]))
        probability = torch.tensor([0.5, 0.5, 0., 0.], dtype=torch.float64)
        self.assertEqual(float(probability @ (alternate[:, 0] - x[:, 0])), 0.)
        self.assertEqual(float(probability @ (old[:, 0] - x[:, 0])), -0.125)
        frozen = copy.deepcopy((x, p))
        enabled = torch.tensor([True, False]); saved_enabled = enabled.clone()
        rng = torch.get_rng_state().clone()
        result = _v435_refine(x, p, enabled, 2, 64)
        self.assertEqual(validate_hif4_params(result, x.shape, 'content'), [])
        values = decode(result, x.shape)
        c = objective(values[:, :64] - x[:, :64], bucket)
        self.assertLess(float(c[0][0]), float(a[0][0]))
        self.assertTrue(torch.equal(c[1], a[1]))
        self.assertTrue(torch.equal(c[2], a[2]))
        self.assertTrue(bool((c[1] <= a[1]).all() and (c[2] <= a[2]).all()))
        for key in p:
            self.assertTrue(torch.equal(result[key][:, 1], p[key][:, 1]))
        assert_tree_equal(self, (x, p), frozen)
        self.assertTrue(torch.equal(enabled, saved_enabled))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_random_parent_lattices_actual_bf16_caps(self):
        parent = load(430, 'content_lattice')
        for rows, width, seed in ((17, 64, 435), (32, 128, 436), (65, 192, 437)):
            generator = torch.Generator().manual_seed(seed)
            x = (torch.randn(rows, width, generator=generator) * 3).to(torch.bfloat16).float()
            noisy = x + torch.randn(rows, width, generator=generator) * 0.15
            p = parent._quantize_linear_values(noisy)
            enabled = torch.arange(width // 64) % 2 == 0
            frozen = copy.deepcopy((x, p, enabled)); rng = torch.get_rng_state().clone()
            result = _v435_refine(x, p, enabled, width // 64, 64)
            self.assertEqual(validate_hif4_params(result, x.shape, 'content'), [])
            before, after = decode(p, x.shape), decode(result, x.shape)
            for head in range(width // 64):
                lo, hi = head * 64, (head + 1) * 64
                if not enabled[head]:
                    for key in p:
                        self.assertTrue(torch.equal(result[key][:, head], p[key][:, head]))
                    continue
                bucket = _v435_buckets(x[:, lo:hi])
                a = objective(before[:, lo:hi] - x[:, lo:hi], bucket)
                b = objective(after[:, lo:hi] - x[:, lo:hi], bucket)
                for left, right in zip(a, b):
                    self.assertTrue(bool((right <= left).all()))
            for key in ('scale_factor', 'scale_lv2', 'scale_lv3'):
                self.assertTrue(torch.equal(result[key], p[key]))
            assert_tree_equal(self, (x, p, enabled), frozen)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_bypasses_and_float32_parameter_dtypes(self):
        x, p = fixture()
        self.assertIs(_v435_refine(x, p, torch.tensor([False, False]), 2, 64), p)
        self.assertIs(_v435_refine(x[:1], p, torch.tensor([True, True]), 2, 64), p)
        self.assertIs(_v435_refine(torch.zeros(4097, 128), p, torch.tensor([True, True]), 2, 64), p)
        self.assertIs(_v435_refine(torch.zeros(2049, 1088), p, torch.ones(17, dtype=torch.bool), 17, 64), p)
        float_params = {key: value.float() for key, value in p.items()}
        result = _v435_refine(x, float_params, torch.tensor([True, False]), 2, 64)
        self.assertEqual(validate_hif4_params(result, x.shape, 'content'), [])
        for key in p:
            self.assertEqual(result[key].dtype, float_params[key].dtype)

    def test_small_lengths_constant_medians_and_saturation(self):
        for rows in (2, 3):
            for mode in ('ties', 'constant', 'positive_limit', 'negative_limit'):
                x, full = fixture()
                x = x[:rows, :64].contiguous()
                p = {key: value[:rows, :1].clone() for key, value in full.items()}
                if mode == 'constant':
                    x.fill_(0.125)
                    p['sign'].zero_(); p['mant'].zero_()
                    self.assertTrue(torch.equal(_v435_buckets(x), torch.zeros(rows, dtype=torch.long)))
                elif mode in ('positive_limit', 'negative_limit'):
                    sign = 1 if mode == 'positive_limit' else -1
                    x.fill_(2 * sign)
                    p['sign'].fill_(sign); p['mant'].fill_(1.75)
                bucket = _v435_buckets(x)
                self.assertTrue(torch.equal(bucket, _v435_buckets(x.clone())))
                a = objective(decode(p, x.shape) - x, bucket)
                result = _v435_refine(x, p, torch.tensor([True]), 1, 64)
                self.assertEqual(validate_hif4_params(result, x.shape, mode), [])
                b = objective(decode(result, x.shape) - x, bucket)
                for before, after in zip(a, b):
                    self.assertTrue(bool((after <= before).all()))
                if mode in ('positive_limit', 'negative_limit'):
                    assert_tree_equal(self, result, p)


if __name__ == '__main__':
    unittest.main(verbosity=2)
