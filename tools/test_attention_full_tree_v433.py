"""Independent scalar enumeration and full-risk checks for v433."""
import copy,json,math,sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.test_v424 import packed_pair
from tools.verify_attention_campaign import equal
from self_check import validate_hif4_params,validate_attention_calibration_result
RECORD={}


class FullTree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)

    def test_independent_full_enumeration(self):
        m=load(433,'enumeration');g=torch.Generator().manual_seed(433)
        optimum=torch.randn(1,64,generator=g)*2
        lower=torch.full_like(optimum,-3.);upper=torch.full_like(optimum,3.)
        sf,lv2,lv3,codes,cost=m._a433_solve_nodes(optimum,lower,upper)
        values=(sf[:,None,None,None]*lv2[:,:,None,None]*lv3[:,:,:,None]*codes*.25).reshape(1,64)
        actual=float((values-optimum).square().sum());best=float('inf');op=optimum[0].tolist()
        # Independent Python scalar enumeration, including all SF and 8-state
        # lv2/lv3 choices for each eight-element subtree.
        for code in range(255):
            scale=(1.+(code&3)/4.)*2.**((code>>2)-48);total=0.
            for block in range(8):
                local=float('inf')
                for a in (1.,2.):
                    for b in (1.,2.):
                        for c in (1.,2.):
                            subtotal=0.
                            for j in range(8):
                                step=scale*a*(b if j<4 else c)*.25
                                lo=max(-7,math.ceil(-3./step));hi=min(7,math.floor(3./step))
                                if lo>hi:subtotal=float('inf');break
                                q=min(hi,max(lo,round(op[block*8+j]/step)))
                                subtotal+=(q*step-op[block*8+j])**2
                            local=min(local,subtotal)
                total+=local
            best=min(best,total)
        self.assertAlmostEqual(actual,best,places=5)
        self.assertTrue(bool(((values>=lower)&(values<=upper)).all()))
        fixed=m._a433_solve_nodes(optimum,values,values)
        fixed_values=(fixed[0][:,None,None,None]*fixed[1][:,:,None,None]*fixed[2][:,:,:,None]*fixed[3]*.25).reshape(1,64)
        self.assertTrue(torch.equal(fixed_values,values))
        RECORD['enumeration']={'sf_count':255,'dp_cost':actual,'scalar_exhaustive_cost':best,'fixed_bounds_exact':True}

    def test_full_risk_and_contract(self):
        m=load(433,'risk');g=torch.Generator().manual_seed(431)
        x=torch.randn(32,128,generator=g)*3
        params=m._quantize_linear_values(x+.8*torch.randn(x.shape,generator=g))
        probability=torch.softmax(torch.randn(2,12,32,generator=g),-1)
        m._ac_parent_probability=lambda *a:probability
        state={'a433_active':torch.tensor([True,False])};before=copy.deepcopy((x,params,state));rng=torch.get_rng_state().clone()
        out=m._a433_tree(x,params,state,2,64)
        self.assertEqual(validate_hif4_params(out,x.shape,'fulltree'),[])
        values=[m._deq_params(p).float().reshape(32,2,64).transpose(0,1).double() for p in (params,out)]
        target=x.reshape(32,2,64).transpose(0,1).double()
        losses=[(probability.double()@(v-target)).square().sum((1,2))+(v-target).square().sum((1,2))/32 for v in values]
        self.assertTrue(bool((losses[1]<=losses[0]+1e-8).all()));self.assertLess(float(losses[1][0]),float(losses[0][0]))
        self.assertTrue(torch.equal(values[0][1],values[1][1]))
        caps=(values[0]-target).square().mean(1).clamp_min(1e-30)
        self.assertTrue(bool(((values[1]-target).square().mean(1)<=caps*1.05+1e-8).all()))
        self.assertTrue(equal(before,(x,params,state)));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        RECORD['risk']={'before':losses[0].tolist(),'after':losses[1].tolist(),'changed':{k:int((params[k]!=out[k]).sum()) for k in params},'all_channel_caps':True}

    def test_real_calibration_and_scope(self):
        p,m=load(422,'parent'),load(433,'real');g=torch.Generator().manual_seed(429)
        samples=[{r:packed_pair(n,128 if r=='q' else 64,g) for r in 'qkv'} for n in (4,8,16,32)]
        original=copy.deepcopy(samples);rng=torch.get_rng_state().clone()
        a=p.hif4_calibration_attention(samples,2,1,64);b=m.hif4_calibration_attention(samples,2,1,64)
        self.assertEqual(validate_attention_calibration_result(b,'fulltree'),[])
        self.assertTrue(equal(a['q_state'],b['q_state']) and equal(a['k_state'],b['k_state']))
        self.assertTrue(equal(original,samples));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for name in ('hif4_calibration_and_quantize_weight','hif4_dynamic_quantize_activation','hif4_dynamic_quantize_q','hif4_dynamic_quantize_k'):
            self.assertEqual(getattr(m,name).__code__.co_code,getattr(p,name).__code__.co_code)
        RECORD['scope']={'qk_calibration_equal':True,'linear_qk_code_equal':True,'input_rng_unchanged':True}


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FullTree))
    (ROOT/'docs/v433-mechanisms.json').write_text(json.dumps({'passed':result.wasSuccessful(),'records':RECORD},indent=2)+'\n')
    sys.exit(not result.wasSuccessful())
