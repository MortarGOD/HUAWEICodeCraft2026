"""Fast deterministic checks for the inverse-population experiment tools."""
import copy
import unittest
import numpy as np
import torch

from fit_attention_feedback import EDGES, fit_mixture, fit_report, simplex_projection, absolute_score_audit
from probe_attention_population import transform_sample
from build_fitted_dataset import quantize_nvfp4
from build_attention_feedback_bank import apportion


class PopulationTests(unittest.TestCase):
    def test_simplex(self):
        for x in (np.array([-10.,1.,3.]), np.zeros(4), np.array([1.])):
            w=simplex_projection(x)
            self.assertTrue((w>=0).all())
            self.assertAlmostEqual(float(w.sum()),1.,places=12)

    def test_integer_group_counts(self):
        result=apportion([.333,.333,.334],50)
        self.assertEqual(int(result.sum()),50)
        self.assertTrue((result>=0).all())
        self.assertGreaterEqual(result[2],result[0])
        with self.assertRaises(ValueError):apportion([-.1,1.1],50)

    def test_known_mixture(self):
        a=np.eye(3)
        b=np.array([.2,.3,.5])
        np.testing.assert_allclose(fit_mixture(a,b,np.ones(3),0),b,atol=1e-8)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            fit_mixture(np.array([[np.nan]]),np.ones(1),np.ones(1))
        np.testing.assert_allclose(fit_mixture(np.zeros((1,2)),np.ones(1),np.ones(1),0),[.5,.5])

    def test_absolute_bound(self):
        self.assertFalse(absolute_score_audit({'310':[.2]*250})['310']['pass'])
        self.assertTrue(absolute_score_audit({'310':[.7]*250})['310']['pass'])

    def test_impossible_target_not_fabricated(self):
        w=fit_mixture(np.array([[1.,2.]]),np.array([-10.]),np.ones(1),0)
        self.assertGreaterEqual(float(np.array([1.,2.])@w),1.)

    def test_heldout_does_not_change_weights(self):
        records=[{'id':str(i),'points':{e:m['target']+i*10 for e,m in EDGES.items()}}
                 for i in range(3)]
        before=fit_report(records)['weights']
        for r in records:
            for e,m in EDGES.items():
                if m['split']=='heldout':r['points'][e]+=100000
        self.assertEqual(before,fit_report(records)['weights'])
        self.assertFalse(fit_report(records)['qualified'])

    def test_input_transform_deterministic_nonmutating(self):
        g=torch.Generator().manual_seed(91)
        sample={r:quantize_nvfp4(torch.randn(16,64,generator=g)) for r in 'qkv'}
        saved=copy.deepcopy(sample)
        args=({'tail':2,'sink':8,'roles':'v'},123,1,1,64)
        a=transform_sample(sample,*args);b=transform_sample(sample,*args)
        for r in 'qkv':
            for left,right,original,current in zip(a[r],b[r],saved[r],sample[r]):
                self.assertTrue(torch.equal(left,right))
                self.assertTrue(torch.equal(original,current))
                self.assertTrue(torch.isfinite(left).all())


if __name__=='__main__':unittest.main()
