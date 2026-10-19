"""Check sequential risk descent under the ORIGINAL, not compounded, cap."""
import copy,json,sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.test_v424 import packed_pair
from tools.verify_attention_campaign import equal
from self_check import validate_hif4_params,validate_attention_calibration_result
RECORD={}


class Hierarchy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)

    def test_two_levels_original_cap_and_shared_probability(self):
        rows=[]
        for seed in (431,432,433,434):
            m=load(434,'hierarchy');g=torch.Generator().manual_seed(seed)
            x=torch.randn(32,128,generator=g)*3
            params=m._quantize_linear_values(x+.8*torch.randn(x.shape,generator=g))
            probability=torch.softmax(torch.randn(2,12,32,generator=g),-1)
            calls=[];stages=[];coarse=m._a433_tree
            m._ac_parent_probability=lambda *a:(calls.append(1) or probability)
            def capture(*a):
                answer=coarse(*a);stages.append(answer);return answer
            m._a433_tree=capture
            state={'a434_active':torch.tensor([True,False])};frozen=copy.deepcopy((x,params,state));rng=torch.get_rng_state().clone()
            result=m._a434_hierarchy(x,params,state,2,64)
            self.assertEqual(len(calls),1);self.assertEqual(len(stages),1)
            values=[m._deq_params(p).float().reshape(32,2,64).transpose(0,1).double() for p in (params,stages[0],result)]
            target=x.reshape(32,2,64).transpose(0,1).double()
            losses=[(probability.double()@(v-target)).square().sum((1,2))+(v-target).square().sum((1,2))/32 for v in values]
            for i in (0,1):self.assertTrue(bool((losses[i+1]<=losses[i]+1e-8).all()))
            self.assertLess(float(losses[2][0]),float(losses[0][0]))
            cap=(values[0]-target).square().mean(1).clamp_min(1e-30)
            for p,v in zip((stages[0],result),values[1:]):
                self.assertEqual(validate_hif4_params(p,x.shape,'hierarchy'),[])
                self.assertTrue(bool(((v-target).square().mean(1)<=cap*1.05+1e-8).all()))
                self.assertTrue(torch.equal(v[1],values[0][1]))
            self.assertTrue(equal((x,params,state),frozen));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            rows.append({'seed':seed,'risk':[a.tolist() for a in losses],'probability_calls':len(calls),'fine_changed_values':int((values[1]!=values[2]).sum())})
        self.assertGreater(sum(r['fine_changed_values'] for r in rows),0)
        RECORD['two_levels']={'rows':rows,'original_cap_at_both_stages':True,'input_state_rng_unchanged':True}

    def test_parent_calibration_and_unmarked_deployment(self):
        p,m=load(422,'parent'),load(434,'scope');g=torch.Generator().manual_seed(429)
        samples=[{r:packed_pair(n,128 if r=='q' else 64,g) for r in 'qkv'} for n in (4,8,16,32)]
        frozen=copy.deepcopy(samples);rng=torch.get_rng_state().clone()
        a=p.hif4_calibration_attention(samples,2,1,64);b=m.hif4_calibration_attention(samples,2,1,64)
        self.assertEqual(validate_attention_calibration_result(b,'hierarchy'),[])
        self.assertTrue(equal(a['q_state'],b['q_state']) and equal(a['k_state'],b['k_state']))
        sample=samples[-1]
        self.assertTrue(equal(p.hif4_dynamic_quantize_v(*sample['v'],1,64,a['v_state']),m.hif4_dynamic_quantize_v(*sample['v'],1,64,a['v_state'])))
        self.assertTrue(equal(frozen,samples));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for name in ('hif4_calibration_and_quantize_weight','hif4_dynamic_quantize_activation','hif4_dynamic_quantize_q','hif4_dynamic_quantize_k'):
            self.assertEqual(getattr(p,name).__code__.co_code,getattr(m,name).__code__.co_code)
        RECORD['scope']={'qk_and_linear_preserved':True,'unmarked_v_preserved':True,'input_rng_unchanged':True}


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Hierarchy))
    (ROOT/'docs/v434-mechanisms.json').write_text(json.dumps({'passed':result.wasSuccessful(),'records':RECORD},indent=2)+'\n')
    sys.exit(not result.wasSuccessful())
