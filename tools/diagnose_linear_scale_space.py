"""Current v422 production-space L1 bound, output propagation, and L4 timing.

The additional exact candidate retains every parent initial candidate and all
subsequent output-aware search/refinement. This is a diagnostic, not a submission.
"""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.linear_exact_tree import exact_tree
from tools.test_v424 import packed_pair
from tools.verify_v420_v421_linear import assert_tree_equal

def block_report(module,x,p,label):
    flat=x.double().reshape(-1,64)
    ix=torch.linspace(0,len(flat)-1,min(256,len(flat))).long()
    target=flat[ix]
    actual=module._linear_score_decode(p).double().reshape(-1,64)[ix]
    t=time.perf_counter();optimal=exact_tree(target);seconds=time.perf_counter()-t
    exact=module._linear_score_decode(optimal).double()
    before=(actual-target).square().sum(-1);after=(exact-target).square().sum(-1)
    relative=torch.where(before>0,1-after/before,torch.zeros_like(before))
    return {'label':label,'blocks':len(ix),'sse_before':float(before.sum()),'sse_exact':float(after.sum()),
            'median_relative_reduction':float(relative.median()),'positive_blocks':int((relative>1e-10).sum()),
            'max_relative_reduction':float(relative.max()),'reference_seconds':seconds,
            'note':'Element SSE of this specified representation; not an AW upper bound.'}

def main():
    torch.set_num_threads(4)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/linear.pt',weights_only=True)
    out={'parent_sha256':hashlib.sha256((ROOT/'solution/solution_v422.py').read_bytes()).hexdigest(),'groups':[],
         'timing_note':'4-thread local diagnostic wall time, no online extrapolation.'}
    for gi in (0,1,9):
        torch.manual_seed(427+gi);m=load(422,f'scale{gi}');g=bank[gi]
        result=m.hif4_calibration_and_quantize_weight(*g['weight'],g['calib_activation_list'])
        st=result['activation_state'];packed=g['test_activation_list'][2]
        target=m.dequantize_nvfp4(*packed).float() @ m.dequantize_nvfp4(*g['weight']).float().T
        record={'group':gi,'shape':list(g['weight'][0].shape),'blocks':[]}
        if st.get('direct_output'):
            calls=[];fn=m._quantize_linear_values
            def capture(x,*a,**kw):
                p=fn(x,*a,**kw);calls.append((x,p));return p
            m._quantize_linear_values=capture
            base=m.hif4_dynamic_quantize_activation(*packed,st)
            for stage,(x,p) in enumerate(calls):
                record['blocks'].append(block_report(m,x,p,f'down_carrier_stage_{stage}'))
            # Replace quantizer only in the direct carrier; keep all residual stages.
            m._quantize_linear_values=lambda x,*a,**kw:exact_tree(x)
            start=time.perf_counter();candidate=m.hif4_dynamic_quantize_activation(*packed,st);extra_time=time.perf_counter()-start
        else:
            teacher=st['unrestricted_target_weight'].float();p=st['unrestricted_continuous_projection'].float()
            effective=torch.linalg.solve(p,teacher.T).T
            record['blocks'].append(block_report(m,effective,result['weight_params'],'up_W_effective_inverse_P_teacher'))
            fn=m._quantize_linear_exact_target;centers=[]
            def capture(x,*a,**kw):
                centers.append(x);return fn(x,*a,**kw)
            m._quantize_linear_exact_target=capture
            base=m.hif4_dynamic_quantize_activation(*packed,st)
            assert len(centers)==1
            record['blocks'].append(block_report(m,centers[0],base,'up_A_final_output_search_vs_continuous_center'))
            # Add, do not replace, the exact SSE start. Existing output search remains.
            source=inspect.getsource(fn)
            needle='    reconstructed = [\n'
            assert source.count(needle)==1
            source=source.replace(needle,'    candidates.append(_research_exact(values))\n'+needle,1)
            m._research_exact=exact_tree
            exec(compile(source,'<L1-output-propagation>','exec'),m.__dict__)
            start=time.perf_counter();candidate=m.hif4_dynamic_quantize_activation(*packed,st);extra_time=time.perf_counter()-start
        b=m._linear_score_decode(result['weight_params'])
        before=(m._linear_score_decode(base)@b.T).float();after=(m._linear_score_decode(candidate)@b.T).float()
        record['propagation']={'mse_before':float((before-target).square().mean()),'mse_after':float((after-target).square().mean()),
                               'changed_output_values':int((before!=after).sum()),'candidate_dynamic_seconds':extra_time,'rows':int(packed[0].shape[0])}
        out['groups'].append(record)
        print(json.dumps(record),flush=True)
    q,s=packed_pair(2048,64,torch.Generator().manual_seed(444))
    s=((torch.arange(s.numel()).reshape_as(s)%13+1)/4).bfloat16()
    raw=m.dequantize_nvfp4(q,s);timing={'general':[],'table':[]}
    for _ in range(3):
        for name in ('general','table'):
            start=time.perf_counter();p=exact_tree(raw,nvfp4=(q,s) if name=='table' else None)
            timing[name].append(time.perf_counter()-start)
    general=exact_tree(raw);table=exact_tree(raw,nvfp4=(q,s));assert_tree_equal(general,table,'lookup equivalence')
    # Actual raw-source ratio-pattern census, streaming weights and all calibration/test A.
    from collections import Counter
    census={'up_W':Counter(),'down_W':Counter(),'A':Counter()}
    for group in bank:
        for role,packed_source in [('up_W' if group['weight'][0].shape[0]>=group['weight'][0].shape[1] else 'down_W',group['weight'])]+[('A',a) for a in group['calib_activation_list']+group['test_activation_list']]:
            sf=packed_source[1].float().reshape(-1,4)
            ratios=sf/sf.amax(-1,keepdim=True).clamp_min(1.e-30)
            modes,counts=torch.unique(ratios,dim=0,return_counts=True)
            census[role].update({tuple(mode):count for mode,count in zip(modes.tolist(),counts.tolist())})
    out['raw_source_ratio_census']={role:{'blocks':sum(count.values()),'distinct_modes':len(count),
        'top_modes':[{'relative_scales':list(key),'count':value} for key,value in count.most_common(10)]} for role,count in census.items()}
    out['L4']={'blocks':2048,'unique_source_scales':13,'seconds':timing,'params_identical':True,
               'production_applicability':'Up W/A are transformed or projected; down carrier is AW residual. None retains raw E2M1 per16 structure. Raw table is not wired into these production paths.'}
    (ROOT/'docs/linear-scale-space-diagnostic.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
