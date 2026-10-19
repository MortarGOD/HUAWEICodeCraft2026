"""Mathematical, final-calibration isolation and legal-scale mechanisms."""
import ast,copy,json,sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.test_v424 import packed_pair
from self_check import validate_hif4_params,validate_attention_calibration_result
from tools.verify_v426 import equal
from tools.attention_token_transfer import frozen_token_step
R={}
class AttentionCampaign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4);cls.modules={v:load(v,f'campaign{v}') for v in (422,429,430,431)}
    def test_parent_calibration_preserved_and_no_input_rng_change(self):
        g=torch.Generator().manual_seed(429)
        samples=[{r:packed_pair(n,128 if r=='q' else 64,g) for r in 'qkv'} for n in (4,8,16,32)]
        frozen=copy.deepcopy(samples);rng=torch.get_rng_state().clone()
        parent=self.modules[422].hif4_calibration_attention(samples,2,1,64)
        for v in (429,430,431):
            m=self.modules[v];result=m.hif4_calibration_attention(samples,2,1,64)
            self.assertEqual(validate_attention_calibration_result(result,str(v)),[])
            for r in ('q','v') if v==429 else ('q','k'):
                self.assertTrue(equal(parent[r+'_state'],result[r+'_state']),(v,r))
            if v==429:
                def strip(st):return {k:strip(x) if isinstance(x,dict) else x for k,x in st.items() if k!='a429_deploy'}
                self.assertTrue(equal(strip(result['k_state']),parent['k_state']))
            self.assertTrue(equal(frozen,samples));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        R['parent_calibration']={'qv_exact_for429':True,'qk_exact_for430_431':True,'429_unmarked_k_exact':True,'state_inputs_rng_unchanged':True}
    def test_temperature_concentration_monotone_and_permutation(self):
        m=self.modules[430];p=torch.softmax(torch.randn(3,8,17,generator=torch.Generator().manual_seed(30)),-1)
        values=[]
        for beta in (.25,1.,2.,4.):
            out=m._a430_power(p,torch.full((3,),beta))
            values.append(out.square().sum(-1).mean(-1))
            torch.testing.assert_close(out.sum(-1),torch.ones(3,8))
            torch.testing.assert_close(out.flip(-1),m._a430_power(p.flip(-1),torch.full((3,),beta)))
        self.assertTrue(bool((torch.stack(values).diff(dim=0)>=-1e-7).all()))
        torch.testing.assert_close(m._a430_power(p,torch.ones(3)),p,rtol=1e-5,atol=1e-7)
        R['temperature']={'normalized':True,'monotone':True,'key_permutation_equivariant':True,'identity_beta':True}
    def test_shared_tree_full_risk_and_bf16_format(self):
        m=self.modules[431];g=torch.Generator().manual_seed(431)
        x=torch.randn(32,128,generator=g)*3
        params=m._quantize_linear_values(x+.8*torch.randn(x.shape,generator=g))
        p=torch.softmax(torch.randn(2,12,32,generator=g),-1)
        old=m._ac_parent_probability;m._ac_parent_probability=lambda *a:p
        state={'a431_active':torch.tensor([True,False])};frozen=copy.deepcopy((params,state,x))
        try:out=m._a431_tree(x,params,state,2,64)
        finally:m._ac_parent_probability=old
        self.assertEqual(validate_hif4_params(out,x.shape,'tree'),[])
        b=m._deq_params(params).reshape(32,2,64).transpose(0,1).double()
        a=m._deq_params(out).reshape(32,2,64).transpose(0,1).double();target=x.reshape(32,2,64).transpose(0,1).double()
        losses=[]
        for y in (b,a):
            e=y-target;losses.append((p.double()@e).square().sum(1)+e.square().sum(1)/32)
        self.assertTrue(bool((losses[1].sum(-1)<=losses[0].sum(-1)+1e-8).all()))
        self.assertTrue(torch.equal(b[1],a[1]));self.assertTrue(torch.equal(params['scale_factor'],out['scale_factor']));self.assertTrue(torch.equal(params['scale_lv2'],out['scale_lv2']))
        changed=int((params['scale_lv3']!=out['scale_lv3']).sum());self.assertGreater(changed,0)
        self.assertTrue(equal((params,state,x),frozen))
        cap=(b-target).square().mean(1).clamp_min(1e-30)
        self.assertTrue(bool(((a-target).square().mean(1)<=cap*1.05+1e-8).all()))
        R['tree']={'lv3_nodes_changed':changed,'risk_before':losses[0].sum(-1).tolist(),'risk_after':losses[1].sum(-1).tolist(),'inactive_head_same':True,'all_channel_mse_caps':True,'scale_factor_lv2_fixed':True}
    def test_full_token_reference_monotone(self):
        g=torch.Generator().manual_seed(45);x=torch.randn(2,8,4,generator=g);step=torch.ones_like(x)*.25;y=x.mul(4).round().clamp(-7,7)/4
        p=torch.softmax(torch.randn(2,5,8,generator=g),-1);h=p.transpose(1,2)@p
        out=frozen_token_step(x,y,step,h)
        H=.7*h/h.diagonal(dim1=-2,dim2=-1).mean(-1)[:,None,None]+.3*torch.eye(8)
        f=lambda a:torch.einsum('htd,htu,hud->h',a-x,H,a-x)
        self.assertTrue(bool((f(out)<=f(y)+1e-6).all()));R['frozen_reference']={'full_token_quadratic_descent':True}
    def test_only_intended_public_scope(self):
        for v in (429,430,431):
            m=self.modules[v];p=self.modules[422]
            for name in ('hif4_calibration_and_quantize_weight','hif4_dynamic_quantize_activation'):
                self.assertEqual(getattr(m,name).__code__.co_code,getattr(p,name).__code__.co_code)
            for role in ('q','v') if v==429 else ('q','k'):
                name='hif4_dynamic_quantize_'+role
                self.assertEqual(getattr(m,name).__code__.co_code,getattr(p,name).__code__.co_code)
        R['scope']={'linear_code_unchanged':True,'independent_parent':422}

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AttentionCampaign))
    (ROOT/'docs/attention-campaign-mechanisms.json').write_text(json.dumps({'passed':result.wasSuccessful(),'records':R},indent=2)+'\n');sys.exit(not result.wasSuccessful())
