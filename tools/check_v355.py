"""Generated-source identity and actual v353/v355 paired regression."""
import ast,contextlib,copy,io,json,time
import torch
from probe_v319_value import ROOT,load_solution
from coverage_factor_source import coverage_source,coverage_checks
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result


def structural_checks():
    actual=(ROOT/'solution/solution_v355.py').read_text()
    expected=(ROOT/'solution/solution_v353.py').read_text()+'\n\n'+coverage_source()
    a=ast.parse(actual);e=ast.parse(expected)
    assert [ast.dump(x) for x in a.body[1:]]==[ast.dump(x) for x in e.body[1:]]
    banned=set('os pathlib io tempfile shutil subprocess socket ctypes pickle marshal importlib zipimport pkgutil runpy mmap resource pty tty sys builtins'.split())
    for node in ast.walk(a):
        if isinstance(node,ast.Import):assert all(x.name.split('.')[0] not in banned for x in node.names)
        if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] not in banned
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):assert node.func.id not in ('open','exec','eval','compile','__import__')
    return 3


def main():
    torch.set_num_threads(4);checks=structural_checks();rows=[]
    modules={v:load_solution(ROOT/f'solution/solution_v{v}.py') for v in (353,355)}
    checks+=coverage_checks(modules[355])
    for dataset,groups in (('local_fitted_models_attention_feedback_v2',(0,1,5,14,19,49)),
                           ('local_fitted_models_covariance_v1',(0,1,24,25,26,49))):
        bank=torch.load(ROOT/dataset/'attn.pt',weights_only=False)
        for gi in groups:
            g=bank[gi];dims=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')];versions={}
            for version,m in modules.items():
                cal=m.hif4_calibration_attention;start=time.perf_counter();state=cal(g['calib'],*dims);seconds=time.perf_counter()-start
                assert not validate_attention_calibration_result(state,f'v{version}');checks+=1
                if version==355 and isinstance(state['v_state'],dict) and 'moment' in state['v_state']:
                    model=state['v_state']['moment'];assert torch.isfinite(model['query_weights']).all()
                    assert torch.allclose(model['query_weights'].sum(1),torch.ones(dims[1]));checks+=2
                m.hif4_calibration_attention=lambda *_,_st=state:copy.deepcopy(_st)
                with contextlib.redirect_stdout(io.StringIO()):scores,_=evaluate_attention(m,[g])
                m.hif4_calibration_attention=cal
                versions[str(version)]={'scores':scores,'calibration_seconds':seconds}
            row={'dataset':dataset,'group':gi,'versions':versions,'delta':sum(versions['355']['scores'])-sum(versions['353']['scores'])}
            rows.append(row);print(row,flush=True)
    report={'complete':True,'checks':checks,'rows':rows,'delta':sum(r['delta'] for r in rows),
            'warning':'Modeled activations; paired local regression, not online score/runtime evidence.'}
    (ROOT/'docs/v355-production-regression.json').write_text(json.dumps(report,indent=2)+'\n');print({'checks':checks,'delta':report['delta']})


if __name__=='__main__':main()
