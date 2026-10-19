"""Actual-source math, active/inactive interface, and official mini checks."""
import argparse,ast,copy,json,os,subprocess,sys,tempfile
import torch
from build_mean_search_versions import ROOT,candidate_source
from probe_mean_search import math_checks
from probe_v319_value import load_solution
from probe_linear_deployed_joint import tree_equal
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_hif4_params,validate_attention_calibration_result


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group()
    group.add_argument('--concentration',action='store_true');group.add_argument('--value-grid',action='store_true')
    group.add_argument('--position-risk',action='store_true');args=parser.parse_args()
    versions=(348,349) if args.position_risk else ((347,) if args.value_grid else ((345,346) if args.concentration else (343,344)))
    build=candidate_source
    if args.concentration:
        from build_concentration_versions import candidate_source as build
    if args.value_grid:
        from build_value_grid_version import candidate_source as build
    if args.position_risk:
        from build_position_versions import candidate_source as build
    torch.set_num_threads(4);rng=torch.Generator().manual_seed(343)
    parent=ast.parse((ROOT/'solution/solution_v324.py').read_text()).body
    banned=set('os pathlib io tempfile shutil subprocess socket ctypes pickle marshal importlib zipimport pkgutil runpy mmap resource pty tty sys builtins'.split())
    report=dict(complete=False,versions={})
    for version in versions:
        source=build(version);assert source==(ROOT/f'solution/solution_v{version}.py').read_text()
        tree=ast.parse(source)
        assert [ast.dump(n) for n in tree.body[1:len(parent)]]==[ast.dump(n) for n in parent[1:]]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):assert all(a.name.split('.')[0] not in banned for a in node.names)
            if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] not in banned
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):assert node.func.id not in ('open','exec','eval','compile','__import__')
        m=load_solution(ROOT/f'solution/solution_v{version}.py');other=load_solution(ROOT/f'solution/solution_v{version}.py')
        if args.position_risk:
            from probe_position_risk import math_checks as position_checks
            math=position_checks(m)
        else:math=dict(checks=0) if args.value_grid else math_checks(m)
        checks=0
        for d in (64,128):
            kh=2
            for t in (1,4,17,129,1025,4097):
                raw=torch.randn(t,kh*d,generator=rng)*.3
                if t in (1,4):raw.zero_()
                packed=quantize_nvfp4(raw)
                penalty=torch.tensor([.01,100.]).reshape(kh,1,1) if args.concentration else .01
                if args.position_risk:
                    anchors=m._PR_ANCHORS
                    probability=torch.randn(kh,5,anchors+4,generator=rng).softmax(-1)
                    gram,kappa=m._pr_moments(probability,anchors);odds,bg=m._pr_odds(probability,anchors)
                    model=dict(length=anchors+4,gram=gram,kappa=kappa,odds=odds,background_kappa=bg)
                    st=dict(position_risk=dict(anchors=anchors,models=[model],rescale=True,active=torch.tensor([True,False]),parent=None))
                elif args.value_grid:
                    st=dict(wide_grid=dict(active=torch.tensor([True,False]),parent=None))
                    saved_grid=m._VG_GRID;m._VG_GRID=m.CAND_GRID
                    assert tree_equal(m._vg_v(*packed,kh,d,None),m._vg_parent_v(*packed,kh,d,None))
                    m._VG_GRID=saved_grid
                    reference=load_solution(ROOT/'solution/solution_v324.py')
                    reference._dyn_table=lambda x,state,has_scale:reference._quantize_weighted(x,torch.ones(1,x.shape[1]),grid=saved_grid)
                    assert tree_equal(m._vg_v(*packed,kh,d,None),reference.hif4_dynamic_quantize_v(*packed,kh,d,None))
                    checks+=2
                else:st=dict(mean_search=dict(penalty=penalty,pairs=m._MS_PAIRS,active=torch.tensor([True,False]),parent=None))
                snapshot=copy.deepcopy((packed,st));random_state=torch.random.get_rng_state().clone()
                assert not validate_attention_calibration_result(dict(q_state=None,k_state=None,v_state=st),'attention')
                out=m.hif4_dynamic_quantize_v(*packed,kh,d,st)
                assert not validate_hif4_params(out,packed[0].shape,'v')
                assert tree_equal(out,other.hif4_dynamic_quantize_v(*packed,kh,d,copy.deepcopy(st)))
                base_function=m._pr_parent_v if args.position_risk else (m._vg_parent_v if args.value_grid else m._ms_parent_v)
                base=base_function(*packed,kh,d,None)
                assert all(torch.equal(p[:,d//64:],base[k][:,d//64:]) for k,p in out.items())
                if t in (1,4097):assert tree_equal(out,base)
                assert tree_equal(snapshot,(packed,st)) and torch.equal(random_state,torch.random.get_rng_state())
                checks+=5
        with tempfile.TemporaryDirectory(prefix=f'hif4-v{version}-') as folder:
            (ROOT.__class__(folder)/'solution.py').write_text(source)
            r=subprocess.run([sys.executable,str(ROOT/'self_check.py'),'--solution_dir',folder,'--datasets_dir',str(ROOT/'mini_sample')],
                cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),capture_output=True,text=True,timeout=240)
        report['versions'][version]=dict(math=math,interface_checks=checks,official_passed=r.returncode==0,stdout=r.stdout,stderr=r.stderr)
        print(version,math['checks'],checks,r.returncode,flush=True);assert r.returncode==0,r.stdout+r.stderr
    report['complete']=True
    tag=f'v{versions[0]}'+(f'-v{versions[1]}' if len(versions)>1 else '')
    (ROOT/f'docs/{tag}-self-check.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
