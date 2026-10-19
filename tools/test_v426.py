"""Synthetic K quotient mathematics, wiring and role contracts before release."""
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
from tools.test_v424 import packed_pair
from self_check import validate_attention_calibration_result, validate_hif4_params
from evaluate_local import gqa_attention

RECORDS = {}


def risk(error, H):
    centered = error - error.mean(1, keepdim=True)
    return (torch.bmm(centered, H) * centered).sum((1, 2))


def without_compaction(module, x, values, unit, H, heads, rounds):
    t, c = x.shape; d = c // heads
    gram = H[torch.arange(heads) * H.shape[0] // heads].float()
    codes = torch.round(values / unit * 4).reshape(t, heads, d).transpose(0, 1).reshape(-1, d).clone()
    step = (.25 * unit).reshape(t, heads, d).transpose(0, 1).reshape(-1, d)
    target = x.reshape(t, heads, d).transpose(0, 1)
    hr = torch.arange(heads * t) // t
    for _ in range(rounds):
        m = torch.bmm((codes * step).reshape(heads, t, d) - target, gram)
        idx, dr, _, _, _, _ = module._v426_quotient_proposals(m.reshape(-1, d), step, codes, hr,
            m.mean(1), gram.reshape(-1, d), gram.diagonal(dim1=-2, dim2=-1), t)
        codes.scatter_add_(1, idx, dr)
    return (codes * step).reshape(heads, t, d).transpose(0, 1).reshape(t, c)


class QuotientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.parent, cls.child = load(422), load(426)

    def test_only_three_k_call_sites_and_no_new_imports(self):
        parent = ast.parse((ROOT/'solution/solution_v422.py').read_text())
        child = ast.parse((ROOT/'solution/solution_v426.py').read_text())
        prefix = ast.Module(body=child.body[:len(parent.body)], type_ignores=[])
        changes = 0
        for node in ast.walk(prefix):
            if isinstance(node, ast.Call):
                before = len(node.keywords)
                node.keywords = [k for k in node.keywords if k.arg != 'quotient']
                changes += before - len(node.keywords)
        self.assertEqual(changes, 3)
        self.assertEqual(ast.dump(parent), ast.dump(prefix))
        suffix = ast.Module(body=child.body[len(parent.body):], type_ignores=[])
        self.assertFalse(any(isinstance(n, (ast.Import, ast.ImportFrom, ast.Global)) for n in ast.walk(suffix)))
        RECORDS['scope'] = {'changed_k_calls': changes, 'other_parent_ast_identical': True, 'new_imports': 0}

    def test_single_and_simultaneous_delta_float64(self):
        rng = torch.Generator().manual_seed(42601); worst = 0.
        for t in (1, 2, 7, 32):
            for _ in range(16):
                a = torch.randn((3, 5, 5), generator=rng, dtype=torch.float64)
                h = a @ a.transpose(1, 2)
                e = torch.randn((3, t, 5), generator=rng, dtype=torch.float64)
                delta = torch.randn(e.shape, generator=rng, dtype=torch.float64)
                if _ % 2 == 0: delta[:, 1:] = 0
                ce = e - e.mean(1, keepdim=True)
                sd = delta.sum(1)
                formula = 2 * (delta * (ce @ h)).sum((1, 2)) + (delta * (delta @ h)).sum((1, 2)) - (sd[:, None] @ h @ sd[:, :, None]).flatten() / t
                direct = risk(e + delta, h) - risk(e, h)
                worst = max(worst, float((formula - direct).abs().max()))
                torch.testing.assert_close(formula, direct, rtol=1e-11, atol=1e-10)
        RECORDS['delta'] = {'cases': 64, 'worst_absolute_float64_error': worst}

    def test_simultaneous_counterexample_and_real_softmax(self):
        x = torch.zeros(2, 64); x[:, 0] = torch.tensor([-.375, .375])
        values = torch.zeros_like(x); unit = torch.full_like(x, 4.); h = torch.eye(64)[None]
        m = -x
        proposal = self.child._v426_quotient_proposals(m, unit / 4, values, torch.zeros(2, dtype=torch.long),
            m.mean(0, keepdim=True), h[0], h.diagonal(dim1=-2, dim2=-1), 2)
        self.assertGreater(float(proposal[-1][0]), 0.)
        self.assertEqual(int((proposal[1] != 0).sum()), 1)
        parent = self.parent._refine_attn_heads(x, values, unit, h, 1, 1)
        child = self.child._refine_attn_heads(x, values, unit, h, 1, 1, quotient=True)
        self.assertTrue(torch.equal(parent, values))
        self.assertFalse(torch.equal(child, parent))
        q = torch.zeros_like(x); q[:, 0] = 8.
        v = torch.zeros_like(x); v[1, 0] = 1.
        ref = gqa_attention(q, x, v, 1, 1, 64)
        before = float((gqa_attention(q.bfloat16(), parent.bfloat16(), v.bfloat16(), 1, 1, 64)-ref).square().sum())
        after = float((gqa_attention(q.bfloat16(), child.bfloat16(), v.bfloat16(), 1, 1, 64)-ref).square().sum())
        self.assertLess(after, before)
        RECORDS['coupled_barrier'] = {'proposed_batch_delta': float(proposal[-1][0]),
            'retained_moves': int((proposal[1] != 0).sum()), 'parent_codes': parent[:, 0].tolist(),
            'child_codes': child[:, 0].tolist(), 'softmax_output_sse_before': before, 'after': after}

    def test_parent_false_boundary_and_random_monotonicity(self):
        rng = torch.Generator().manual_seed(42602)
        comparisons = []
        for t in (0, 1, 2, 8, 33):
            for b in (1, 4):
                x = torch.randn(t, b*64, generator=rng)
                unit = torch.ones_like(x)*2
                values = (x/.5).round().clamp(-7, 7)*.5
                a = torch.randn(2 if b==4 else 1, 64, 64, generator=rng)
                h = a @ a.transpose(1, 2) / 64 + torch.eye(64)[None]*.1
                inputs = copy.deepcopy((x, values, unit, h)); state = torch.get_rng_state().clone()
                expected = self.parent._refine_attn_heads(x, values, unit, h, b, 1)
                actual = self.child._refine_attn_heads(x, values, unit, h, b, 1, quotient=False)
                self.assertTrue(torch.equal(expected, actual))
                out = self.child._refine_attn_heads(x, values, unit, h, b, 1, quotient=True)
                assert_tree_equal(self, inputs, (x, values, unit, h))
                self.assertTrue(torch.equal(state, torch.get_rng_state()))
                self.assertTrue(torch.isfinite(out).all())
                self.assertTrue(((out/(unit*.25)).abs()<=7).all())
                if t<=1:self.assertTrue(torch.equal(out, values));continue
                hh = h[torch.arange(b)*h.shape[0]//b].double()
                loss = lambda v: risk((v-x).reshape(t,b,64).transpose(0,1).double(),hh)
                before,after=loss(values),loss(out)
                self.assertTrue((after<=before+1e-5*before.clamp_min(1)).all())
                comparisons.append([t,b,float(before.sum()),float(after.sum())])
        RECORDS['random_risk_and_false_path'] = comparisons

    def test_frozen_rows_against_rechecked_reference(self):
        records=[]
        for seed in range(20):
            rng=torch.Generator().manual_seed(426100+seed)
            x=torch.randn(8,8,generator=rng); unit=torch.ones_like(x)*4
            values=x.round().clamp(-7,7)
            a=torch.randn(1,8,8,generator=rng);h=a@a.transpose(1,2)/8+torch.eye(8)[None]*.1
            compact=self.child._refine_attn_heads(x,values,unit,h,1,1,total_rounds=8,quotient=True)
            full=without_compaction(self.child,x,values,unit,h,1,8)
            costs=[float(risk((z-x)[None].double(),h.double())) for z in (values,compact,full)]
            records.append({'seed':426100+seed,'before':costs[0],'compressed':costs[1],'rechecked':costs[2]})
        RECORDS['compression_limit']={'all_20_seeds':records,'note':'Same budget without freezing, not a global optimum or a general error bound.'}

    def test_explicit_role_dispatch_forced_paths(self):
        rng=torch.Generator().manual_seed(42603)
        pair=packed_pair(16,64,rng)
        state={'kvh':1,'rot':False,'u':torch.eye(64)[None],'H':torch.eye(64)[None],'rf':1}
        original=self.child._refine_attn_heads;calls=[]
        def traced(*args,**kw):
            calls.append(bool(kw.get('quotient',False)));return original(*args,**kw)
        with patch.object(self.child,'_refine_attn_heads',traced):
            for role in ('q','k'):
                self.child._dyn_qk(*pair,state,1,64,role=role)
                self.child._kt_fixed_grid(pair[0].float(),state,1,64,role)
        self.assertEqual(calls,[False,False,True,True])
        for role in ('q','k'):
            result=self.child._dyn_qk(*pair,state,1,64,role=role)
            self.assertEqual(validate_hif4_params(result,pair[0].shape,role),[])
        RECORDS['forced_role_routes']=calls

    def test_three_role_calibration_and_fresh_module_replay(self):
        rng=torch.Generator().manual_seed(42604);cal=[]
        for t in (8,16,24,32,48):
            cal.append({role:packed_pair(t,128 if role=='q' else 64,rng) for role in 'qkv'})
        frozen=copy.deepcopy(cal);role_states={}
        for role in 'qkv':
            sample=copy.deepcopy(cal)
            for entry in sample:
                for peer in 'qkv':
                    if peer!=role:
                        entry[peer]=(entry[peer][0].flip(0),entry[peer][1].clone())
            state=self.child.hif4_calibration_attention(sample,2,1,64)
            self.assertEqual(validate_attention_calibration_result(state,'synthetic-role'),[])
            role_states[role]=state[role+'_state']
        test={r:packed_pair(24,128 if r=='q' else 64,rng) for r in 'qkv'}
        snapshot=copy.deepcopy(role_states);saved=torch.get_rng_state().clone()
        for role in 'vkq':
            args=(*test[role],2 if role=='q' else 1,64,role_states[role])
            first=getattr(self.child,'hif4_dynamic_quantize_'+role)(*args)
            fresh=load(426,role)
            second=getattr(fresh,'hif4_dynamic_quantize_'+role)(*copy.deepcopy(args))
            assert_tree_equal(self,first,second)
            self.assertEqual(validate_hif4_params(first,test[role][0].shape,role),[])
        assert_tree_equal(self,snapshot,role_states);assert_tree_equal(self,frozen,cal)
        self.assertTrue(torch.equal(saved,torch.get_rng_state()))
        RECORDS['role_isolation']={'three_separate_calibrations':True,'fresh_reversed_role_replay':True,
            'input_state_rng_preserved':True,'limitation':'Shuffled dummy peers are a stress assumption; actual online dummy source is unknown.'}


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(QuotientTests))
    RECORDS['tests_run']=result.testsRun;RECORDS['passed']=result.wasSuccessful()
    RECORDS['source_sha256']=hashlib.sha256((ROOT/'solution/solution_v426.py').read_bytes()).hexdigest()
    (ROOT/'docs/v426-mechanisms.json').write_text(json.dumps(RECORDS,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
