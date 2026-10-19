"""Structural and actual-file regression checks for v354."""
import ast,contextlib,copy,io,json,time
import torch
from probe_v319_value import ROOT,load_solution
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result


def structural_checks():
    a=ast.parse((ROOT/'solution/solution_v353.py').read_text())
    b=ast.parse((ROOT/'solution/solution_v354.py').read_text())
    assert len(a.body)==len(b.body)
    changed=[(i,getattr(y,'name',type(y).__name__)) for i,(x,y) in enumerate(zip(a.body[1:],b.body[1:]),1) if ast.dump(x)!=ast.dump(y)]
    assert len(changed)==1 and changed[0][1]=='hif4_calibration_attention',changed
    banned=set('os pathlib io tempfile shutil subprocess socket ctypes pickle marshal importlib zipimport pkgutil runpy mmap resource pty tty sys builtins'.split())
    for node in ast.walk(b):
        if isinstance(node,ast.Import):assert all(x.name.split('.')[0] not in banned for x in node.names)
        if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] not in banned
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):assert node.func.id not in ('open','exec','eval','compile','__import__')
    return 3,changed


def main():
    torch.set_num_threads(4);checks,changed=structural_checks();rows=[]
    modules={v:load_solution(ROOT/f'solution/solution_v{v}.py') for v in (353,354)}
    for dataset,groups in (('local_fitted_models_attention_feedback_v2',(0,1,5,14,19,49)),
                           ('local_fitted_models_covariance_v1',(0,1,24,25,26,49))):
        bank=torch.load(ROOT/dataset/'attn.pt',weights_only=False)
        for gi in groups:
            g=bank[gi];dims=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')];versions={}
            for version,m in modules.items():
                cal=m.hif4_calibration_attention;start=time.perf_counter();state=cal(g['calib'],*dims);elapsed=time.perf_counter()-start
                assert not validate_attention_calibration_result(state,f'v{version}');checks+=1
                debug=copy.deepcopy(m.JOINT_DEBUG);assert not debug.get('error')
                m.hif4_calibration_attention=lambda *_,_st=state:copy.deepcopy(_st)
                with contextlib.redirect_stdout(io.StringIO()):scores,_=evaluate_attention(m,[g])
                m.hif4_calibration_attention=cal
                versions[str(version)]={'scores':scores,'calibration_seconds':elapsed,'joint_debug':debug}
            row={'dataset':dataset,'group':gi,'versions':versions,'delta':sum(versions['354']['scores'])-sum(versions['353']['scores'])}
            rows.append(row);print(row,flush=True)
    report={'complete':True,'checks':checks,'changed':changed,'rows':rows,'delta':sum(r['delta'] for r in rows),
            'warning':'Modeled activations, paired local regression; not an online score or runtime guarantee.'}
    (ROOT/'docs/v354-production-regression.json').write_text(json.dumps(report,indent=2)+'\n');print({'checks':checks,'delta':report['delta']})


if __name__=='__main__':main()
