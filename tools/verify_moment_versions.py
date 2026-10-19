"""Actual-file score/state regression, followed by isolated interface timings."""
import argparse,contextlib,copy,io,json,time
import torch
from build_moment_versions import ROOT,candidate_source
from probe_v319_value import load_solution
from probe_linear_deployed_joint import tree_equal
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result,validate_hif4_params


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group()
    group.add_argument('--mean-search',action='store_true');group.add_argument('--concentration',action='store_true')
    group.add_argument('--value-grid',action='store_true');group.add_argument('--position-risk',action='store_true')
    group.add_argument('--factor',action='store_true')
    parser.add_argument('--all-groups',action='store_true')
    parser.add_argument('--dataset',choices=('feedback','covariance'),default='feedback');args=parser.parse_args()
    versions=(348,349) if args.position_risk else ((347,) if args.value_grid else ((345,346) if args.concentration else ((343,344) if args.mean_search else (341,342))))
    if args.factor:versions=(350,)
    build=candidate_source
    if args.mean_search:
        from build_mean_search_versions import candidate_source as build
    if args.concentration:
        from build_concentration_versions import candidate_source as build
    if args.value_grid:
        from build_value_grid_version import candidate_source as build
    if args.position_risk:
        from build_position_versions import candidate_source as build
    guard_name='_ms_guard' if args.mean_search or args.concentration else '_am_guard'
    debug_name='MS_DEBUG' if args.mean_search or args.concentration else 'AM_DEBUG'
    if args.value_grid:guard_name='_vg_guard';debug_name='VG_DEBUG'
    if args.position_risk:guard_name='_pr_guard';debug_name='PR_DEBUG'
    torch.set_num_threads(4)
    dataset='local_fitted_models_attention_feedback_v2' if args.dataset=='feedback' else 'local_fitted_models_covariance_v1'
    bank=torch.load(ROOT/dataset/'attn.pt',weights_only=False)
    modules={v:load_solution(ROOT/f'solution/solution_v{v}.py') for v in (324,*versions)}
    for v in versions:assert (ROOT/f'solution/solution_v{v}.py').read_text()==build(v)
    rows=[]
    selected=range(len(bank)) if args.all_groups else ((0,1,5,14,19,49) if args.dataset=='feedback' else (0,1,24,25,26,49))
    for gi in selected:
        g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        start=time.perf_counter();root=modules[324].hif4_calibration_attention(g['calib'],qh,kh,d);cal=time.perf_counter()-start
        states={324:root};extra={324:0.};debug={};snapshot=copy.deepcopy(root)
        for v in versions:
            start=time.perf_counter();states[v]=getattr(modules[v],guard_name)(g['calib'],root,qh,kh,d);extra[v]=time.perf_counter()-start
            debug[v]=copy.deepcopy(getattr(modules[v],debug_name))
            assert not debug[v]['error'] and tree_equal(snapshot,root)
            assert not validate_attention_calibration_result(states[v],'attention')
            for k in ('q_state','k_state'):assert tree_equal(root[k],states[v][k])
        common=0.;vt={v:0. for v in modules}
        for s in g['test']:
            for role in 'qk':
                start=time.perf_counter();getattr(modules[324],'hif4_dynamic_quantize_'+role)(*s[role],qh if role=='q' else kh,d,root[role+'_state']);common+=time.perf_counter()-start
            for v in (tuple(modules) if gi%2==0 else tuple(reversed(modules))):
                start=time.perf_counter();out=modules[v].hif4_dynamic_quantize_v(*s['v'],kh,d,states[v]['v_state']);vt[v]+=time.perf_counter()-start
                assert not validate_hif4_params(out,s['v'][0].shape,'v')
        scores={}
        if args.value_grid:
            vg=modules[347];grid=vg._VG_GRID
            reference=load_solution(ROOT/'solution/solution_v324.py')
            reference._dyn_table=lambda x,state,has_scale:reference._quantize_weighted(x,torch.ones(1,x.shape[1]),grid=grid)
            for sample in g['test']:
                vg._VG_GRID=vg.CAND_GRID
                assert tree_equal(vg._vg_v(*sample['v'],kh,d,root['v_state']),vg._vg_parent_v(*sample['v'],kh,d,root['v_state']))
                vg._VG_GRID=grid
                assert tree_equal(vg._vg_v(*sample['v'],kh,d,root['v_state']),reference.hif4_dynamic_quantize_v(*sample['v'],kh,d,root['v_state']))
            vg._VG_GRID=grid
        for v in modules:
            evaluator=load_solution(ROOT/f'solution/solution_v{v}.py')
            evaluator.hif4_calibration_attention=lambda *_,_st=states[v]:copy.deepcopy(_st)
            with contextlib.redirect_stdout(io.StringIO()):scores[v],_=evaluate_attention(evaluator,[g])
        rows.append(dict(group=gi,source_key=g.get('key'),
            kv_equal_samples=sum(all(torch.equal(s['k'][j],s['v'][j]) for j in (0,1)) for phase in ('calib','test') for s in g[phase]),
            scores=scores,debug=debug,delta={v:sum(scores[v])-sum(scores[324]) for v in versions},
            timing=dict(base_calibration=cal,common_qk=common,extra_calibration=extra,value=vt,total={v:cal+common+extra[v]+vt[v] for v in modules})))
        print(gi,rows[-1]['delta'],rows[-1]['timing']['total'],flush=True)
    screen_groups={0,1,5,14,19,49}
    outside=[r for r in rows if r['group'] not in screen_groups]
    report=dict(complete=True,dataset=dataset,group_count=len(rows),test_count=sum(len(r['scores'][324]) for r in rows),
        outside_screen_delta=({v:sum(r['delta'][v] for r in outside) for v in versions} if args.dataset=='feedback' else None),
        case_counts={v:{name:sum(1 for r in rows for a,b in zip(r['scores'][v],r['scores'][324]) if predicate(a-b))
                      for name,predicate in [('up',lambda x:x>1e-12),('down',lambda x:x< -1e-12),('same',lambda x:abs(x)<=1e-12)]} for v in versions},
        rows=rows,delta={v:sum(r['delta'][v] for r in rows) for v in versions},
        totals={v:sum(r['timing']['total'][v] for r in rows) for v in modules},
        warning='See group_count/test_count for scope. Checkpoint-derived weights with modeled activations, not real model forward activations. Shared-parent incremental CPU interface timing, not full independent runs; no online score/time guarantee.')
    tag=f'v{versions[0]}'+(f'-v{versions[1]}' if len(versions)>1 else '')
    suffix='-full' if args.all_groups else ''
    if args.dataset!='feedback':suffix+='-'+args.dataset
    (ROOT/f'docs/{tag}-production-regression{suffix}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['delta'],report['totals'],flush=True)


if __name__=='__main__':main()
