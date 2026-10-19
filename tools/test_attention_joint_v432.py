"""v432 joint-risk binding, legal-grid descent and one-guard calibration tests."""
import copy
import json
import sys
import unittest
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.test_v424 import packed_pair
from tools.verify_attention_campaign import equal
from self_check import validate_hif4_params,validate_attention_calibration_result

RECORD={}


class JointAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_consistent_risk_and_identity_temperature(self):
        a,b=load(431,'treeparent'),load(432,'joint')
        g=torch.Generator().manual_seed(432)
        x=torch.randn(32,128,generator=g)*3
        params=a._quantize_linear_values(x+.8*torch.randn(x.shape,generator=g))
        p=torch.softmax(torch.randn(2,12,32,generator=g),-1)
        a._ac_parent_probability=lambda *args:p
        b._ac_parent_probability=lambda *args:p
        state={'a431_active':torch.tensor([True,False]),'a430_active':torch.tensor([True,False]),'a430_temperature':torch.ones(2)}
        self.assertTrue(equal(a._a431_tree(x,params,state,2,64),b._a431_tree(x,params,state,2,64)))
        state['a430_temperature']=torch.tensor([2.,1.]);before=copy.deepcopy((x,params,state));rng=torch.get_rng_state().clone()
        out=b._a431_tree(x,params,state,2,64)
        self.assertEqual(validate_hif4_params(out,x.shape,'joint'),[])
        probability=b._vk_probabilities(x,state,2).double()
        target=x.reshape(32,2,64).transpose(0,1).double()
        values=[b._deq_params(v).reshape(32,2,64).transpose(0,1).double() for v in (params,out)]
        loss=[(probability@(v-target)).square().sum((1,2))+(v-target).square().sum((1,2))/32 for v in values]
        self.assertTrue(bool((loss[1]<=loss[0]+1e-8).all()))
        changed=int((params['scale_lv3']!=out['scale_lv3']).sum());self.assertGreater(changed,0)
        self.assertTrue(torch.equal(values[0][1],values[1][1]))
        self.assertTrue(equal(before,(x,params,state)));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        cap=(values[0]-target).square().mean(1).clamp_min(1e-30)
        self.assertTrue(bool(((values[1]-target).square().mean(1)<=cap*1.05+1e-8).all()))
        RECORD['joint_risk']={'before':loss[0].tolist(),'after':loss[1].tolist(),'changed_nodes':changed,'beta1_equals_v431':True}

    def test_one_parent_and_one_joint_guard(self):
        m=load(432,'oneguard');calls=[]
        parent={'q_state':None,'k_state':None,'v_state':{'active':torch.tensor([True,True])}}
        m._ac_parent_calibration=lambda *a:(calls.append('parent') or copy.deepcopy(parent))
        m._ac_eligible=lambda *a:True
        m._a430_fit=lambda *a:torch.tensor([1.4,1.6])
        def guard(hold,base,trial,*a):
            calls.append('guard');self.assertIn('a430_temperature',trial);self.assertIn('a431_active',trial)
            return torch.tensor([True,False])
        m._ac_guard=guard
        result=m.hif4_calibration_attention([{}],4,2,64)
        self.assertEqual(calls,['parent','guard'])
        self.assertTrue(torch.equal(result['v_state']['a430_active'],result['v_state']['a431_active']))
        self.assertEqual(validate_attention_calibration_result(result,'joint'),[])
        RECORD['guard']={'parent_calls':1,'joint_guard_calls':1,'joint_head_mask':True}

    def test_real_parent_calibration_and_scope(self):
        parent,m=load(422,'reference'),load(432,'real')
        g=torch.Generator().manual_seed(429)
        samples=[{r:packed_pair(n,128 if r=='q' else 64,g) for r in 'qkv'} for n in (4,8,16,32)]
        original=copy.deepcopy(samples);rng=torch.get_rng_state().clone()
        a=parent.hif4_calibration_attention(samples,2,1,64);b=m.hif4_calibration_attention(samples,2,1,64)
        self.assertEqual(validate_attention_calibration_result(b,'joint'),[])
        self.assertTrue(equal(a['q_state'],b['q_state']) and equal(a['k_state'],b['k_state']))
        self.assertTrue(equal(original,samples));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for name in ('hif4_calibration_and_quantize_weight','hif4_dynamic_quantize_activation','hif4_dynamic_quantize_q','hif4_dynamic_quantize_k'):
            self.assertEqual(getattr(parent,name).__code__.co_code,getattr(m,name).__code__.co_code)
        sample=samples[0]
        self.assertTrue(equal(parent.hif4_dynamic_quantize_v(*sample['v'],1,64,a['v_state']),m.hif4_dynamic_quantize_v(*sample['v'],1,64,a['v_state'])))
        RECORD['scope']={'unmarked_v_equals_parent':True,'qk_calibration_equal':True,'linear_qk_code_equal':True,'input_rng_unchanged':True}


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(JointAttention))
    (ROOT/'docs/v432-mechanisms.json').write_text(json.dumps({'passed':result.wasSuccessful(),'records':RECORD},indent=2)+'\n')
    sys.exit(not result.wasSuccessful())
