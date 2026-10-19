"""L1-L4 mathematical, deployment-state and mechanism tests on synthetic inputs."""
import ast
import copy
import json
from pathlib import Path
import sys
import unittest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.linear_exact_tree import exact_tree, scales, tree_choice
from tools.test_v422 import load
from tools.test_v424 import packed_pair
from self_check import validate_hif4_params,validate_frozen_state
from tools.verify_v420_v421_linear import assert_tree_equal

RECORD={}
class Campaign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.p=load(422,'campaign');cls.j=load(427,'campaign');cls.r=load(428,'campaign')
    def test_exact_tree_independent_exhaustive_scale_and_local_assignments(self):
        g=torch.Generator().manual_seed(427)
        x=torch.randn(5,64,generator=g).double()*3
        x[1]*=2**-49;x[2]*=40000;x[3]=0
        p=exact_tree(x)
        self.assertEqual(validate_hif4_params(p,x.shape,'exact'),[])
        actual=self.p._linear_score_decode(p).double()
        best=torch.full((5,),torch.inf,dtype=torch.float64)
        codes=torch.arange(-7,8,dtype=torch.float64)
        for sf in scales():
            # Independent direct code enumeration and all 8 lv2/lv3 assignments per octet.
            local=[]
            for l2 in (1,2):
                errs=[]
                for l3 in (1,2):
                    vals=(codes/4*l2*l3*sf).bfloat16().double()
                    errs.append((x[:,:,None]-vals).square().min(-1).values.reshape(5,8,2,4).sum(-1))
                for a,b in ((0,0),(0,1),(1,0),(1,1)):
                    local.append(errs[a][:,:,0]+errs[b][:,:,1])
            err=torch.stack(local).min(0).values.sum(-1)
            best=torch.minimum(best,err)
        loss=(x-actual).square().sum(-1)
        torch.testing.assert_close(loss,best,rtol=1e-13,atol=1e-30)
        RECORD['L1']={'scales':255,'octet_assignments':8,'blocks':5,'max_relative_error':float(((loss-best).abs()/best.clamp_min(1e-100)).max())}
    def test_nvfp4_table_general_equivalence_and_structure_rejection(self):
        q,s=packed_pair(200,64,torch.Generator().manual_seed(428))
        s=(torch.arange(s.numel()).reshape_as(s)%13+1).bfloat16()/4
        raw=self.p.dequantize_nvfp4(q,s)
        general=exact_tree(raw);table=exact_tree(raw,nvfp4=(q,s))
        assert_tree_equal(general,table,'L4')
        with self.assertRaises(ValueError):exact_tree(raw.float()+.0123,nvfp4=(q,s))
        # Identical four scales do not fix the optimum: level distribution matters.
        a=torch.zeros(1,64,dtype=torch.bfloat16);b=a.clone();b[0,0]=6
        pa=exact_tree(a);pb=exact_tree(b)
        self.assertFalse(torch.equal(pa['scale_factor'],pb['scale_factor']))
        RECORD['L4']={'equivalent_blocks':200,'transformed_input_rejected':True,'same_scales_different_distributions':True}
    def test_rotation_orthogonality_and_continuous_aw(self):
        grad=torch.randn(2,64,64,generator=torch.Generator().manual_seed(2))
        r=self.r._lr_givens(grad,.08)
        torch.testing.assert_close(r @ r.transpose(-1,-2),torch.eye(64).expand_as(r),rtol=1e-6,atol=1e-7)
        a=torch.randn(9,128);b=torch.randn(256,128)
        ar=torch.einsum('ngi,gij->ngj',a.reshape(-1,2,64),r).reshape_as(a)
        br=torch.einsum('ogi,gij->ogj',b.reshape(-1,2,64),r).reshape_as(b)
        torch.testing.assert_close(ar@br.T,a@b.T,rtol=1e-4,atol=1e-5)
        RECORD['L3_rotation']={'pairs_per_block':8,'orthogonal':True,'continuous_aw_preserved':True}
    def test_same_w_recompile_and_actual_legal_rotation(self):
        g=torch.Generator().manual_seed(42730)
        w=packed_pair(128,64,g);cal=[packed_pair(n,64,g) for n in (2,4,8,12,16)]
        parent=self.j._linear_research_parent_calibration(*w,cal)
        frozen=copy.deepcopy(parent);rng=torch.get_rng_state().clone()
        st=parent['activation_state'];p=st['unrestricted_continuous_projection']
        rebuilt=self.j._lr_compile(st,parent['weight_params'],p)
        for key in ('unrestricted_deployed_weight','linear_full_gram','unrestricted_continuous_projection','unrestricted_block_gram','unrestricted_block_inverse','unrestricted_micro_inverse'):
            torch.testing.assert_close(st[key],rebuilt[key],rtol=0,atol=0)
        for m in (self.j,self.r):
            z=m.hif4_dynamic_quantize_activation(*cal[2],rebuilt)
            assert_tree_equal(z,m.hif4_dynamic_quantize_activation(*cal[2],st),'same-W actual A')
        teacher=self.j.dequantize_nvfp4(*w).float()
        losses,dec=self.r._lr_evaluate(parent,cal[:4],teacher)
        grad=self.r._lr_rotation_gradient(parent,dec)
        trial=self.r._lr_rotation_trial(parent,self.r._lr_givens(grad,.08))
        self.assertEqual(validate_hif4_params(trial['weight_params'],w[0].shape,'rotation'),[])
        changed=int((self.r._linear_score_decode(trial['weight_params'])!=self.r._linear_score_decode(parent['weight_params'])).sum())
        self.assertGreater(changed,0)
        a,ad=self.r._lr_evaluate(trial,cal[:4],teacher)
        self.assertTrue(torch.isfinite(a).all())
        self.assertEqual(validate_frozen_state(trial['activation_state'],'rotation-state'),[])
        assert_tree_equal(parent,frozen,'parent unchanged')
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        RECORD['deployment']={'same_w_all_grams_p_and_actual_A_equal':True,'rotation_changed_W_values':changed,'parent_fold_loss':losses.tolist(),'trial_fold_loss':a.tolist(),'state_and_rng_unchanged':True}
    def test_joint_proposal_and_all_projection_variants_reencode_actual_a(self):
        g=torch.Generator().manual_seed(42740)
        w=packed_pair(128,64,g);cal=[packed_pair(n,64,g) for n in (2,4,8,12,16)]
        parent=self.j._linear_research_parent_calibration(*w,cal)
        teacher=self.j.dequantize_nvfp4(*w).float()
        before,dec=self.j._lr_evaluate(parent,cal[:4],teacher)
        params=self.j._lr_joint_proposal(parent,dec)
        b=self.j._linear_score_decode(params).float()
        st=parent['activation_state'];p0=st['unrestricted_continuous_projection']
        gram=b.T@b;ridge=st['unrestricted_projection_ridge']*gram.diagonal().mean()
        gram.diagonal().add_(ridge)
        p1=torch.linalg.solve(gram,(st['unrestricted_target_weight'].float().T@b).T).T
        losses=[];changed=[]
        for f in (0.,.25,1.):
            trial=dict(parent,weight_params=params,activation_state=self.j._lr_compile(st,params,p0+f*(p1-p0)))
            loss,out=self.j._lr_evaluate(trial,cal[:4],teacher)
            losses.append(loss.tolist())
            changed.append(sum(int((a[1]!=b[1]).sum()) for a,b in zip(dec,out)))
        self.assertGreater(int((b!=self.j._linear_score_decode(parent['weight_params'])).sum()),0)
        self.assertTrue(any(changed))
        RECORD['L2_joint']={'before':before.tolist(),'fractions':[0,.25,1],'actual_reencoded_losses':losses,'changed_A_values':changed}

    def test_scope_and_budgets(self):
        prefix=(ROOT/'solution/solution_v422.py').read_bytes()
        for v,m in ((427,self.j),(428,self.r)):
            raw=(ROOT/f'solution/solution_v{v}.py').read_bytes();self.assertTrue(raw.startswith(prefix))
            tree=ast.parse(raw[len(prefix):].decode())
            self.assertFalse(any(isinstance(n,(ast.Import,ast.ImportFrom,ast.Global)) for n in ast.walk(tree)))
            for name in ('hif4_dynamic_quantize_activation','hif4_calibration_attention','hif4_dynamic_quantize_q','hif4_dynamic_quantize_k','hif4_dynamic_quantize_v'):
                self.assertEqual(getattr(m,name).__code__.co_code,getattr(self.p,name).__code__.co_code)
            self.assertIsNone(m._lr_inputs({'activation_state':{'direct_output':True}},torch.empty(64,128),None,[None]*5))
            self.assertIsNone(m._lr_inputs({'activation_state':{}},torch.empty(8192,4096,device='meta'),None,[None]*5))
        RECORD['scope']={'versions':[427,428],'v422_prefix':True,'attention_and_dynamic_code_unchanged':True,'overbudget_skips':True}

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Campaign)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    (ROOT/'docs/linear-campaign-mechanisms.json').write_text(json.dumps({'passed':result.wasSuccessful(),'records':RECORD},indent=2)+'\n')
    sys.exit(not result.wasSuccessful())
