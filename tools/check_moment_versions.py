"""Mathematical invariants, forced-active interfaces, and official mini validation."""
import argparse,ast,copy,json,os,subprocess,sys,tempfile
import torch
from build_moment_versions import ROOT,candidate_source
from probe_v319_value import load_solution
from probe_linear_deployed_joint import tree_equal
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_hif4_params,validate_attention_calibration_result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--factor',action='store_true');parser.add_argument('--cal-cache',action='store_true');args=parser.parse_args()
    args.factor=args.factor or args.cal_cache
    torch.set_num_threads(4);rng=torch.Generator().manual_seed(341)
    parent_ast=ast.parse((ROOT/'solution/solution_v324.py').read_text()).body
    banned=set('os pathlib io tempfile shutil subprocess socket ctypes pickle marshal importlib zipimport pkgutil runpy mmap resource pty tty sys builtins'.split())
    report=dict(complete=False,versions={})
    for version in ((351,) if args.cal_cache else ((350,) if args.factor else (341,342))):
        source=candidate_source(version)
        assert source==(ROOT/f'solution/solution_v{version}.py').read_text()
        tree=ast.parse(source)
        assert [ast.dump(n) for n in tree.body[1:len(parent_ast)]]==[ast.dump(n) for n in parent_ast[1:]]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):assert all(a.name.split('.')[0] not in banned for a in node.names)
            if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] not in banned
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):assert node.func.id not in ('open','exec','eval','compile','__import__')
        m=load_solution(ROOT/f'solution/solution_v{version}.py')
        other=load_solution(ROOT/f'solution/solution_v{version}.py');checks=0
        if args.factor:
            from probe_factor_risk import math_checks
            checks+=math_checks(m)
        if args.cal_cache:
            from check_factor_cal_cache import cache_checks
            checks+=cache_checks(m)
        for t in (1,4,17,65):
            x=torch.randn(2,t,8,generator=rng,dtype=torch.float64)
            step=torch.rand(2,t,8,generator=rng,dtype=torch.float64)+.1
            base=(x/step).round().clamp(-7,7)*step
            a=torch.softmax(torch.randn(2,t,generator=rng,dtype=torch.float64),1)
            variance=torch.rand(2,t,generator=rng,dtype=torch.float64)*.1/t
            gradient=torch.randn(2,t,8,generator=rng,dtype=torch.float64)*.01/t
            h=a[:,:,None]*a[:,None,:]+torch.diag_embed(variance)
            assert bool((torch.linalg.eigvalsh(h)>-1e-12).all())
            def cost(value):
                e=value-x
                return (e*(h@e)).sum()+2*(e*gradient).sum()
            error=base-x
            proxy=(a[:,:,None]*error).sum(1).square().sum()+(variance[:,:,None]*error.square()).sum()+2*(error*gradient).sum()
            assert torch.allclose(proxy,cost(base),atol=1e-10,rtol=1e-10)
            delta=torch.zeros_like(base);delta[:,0,0]=.03
            exact=(2*(h@error+gradient)*delta+delta*(h@delta)).sum()
            assert torch.allclose(cost(base+delta)-cost(base),exact,atol=1e-10,rtol=1e-10)
            previous=cost(base)
            for rounds in (1,2,8,16):
                result=m._am_search(x,base,step,a,variance,gradient,rounds)
                assert cost(result)<=previous+1e-10
                assert torch.allclose(result/step,(result/step).round(),atol=1e-10,rtol=0)
                assert bool(((result/step).abs()<=7+1e-10).all())
                previous=cost(result);checks+=3
            checks+=3
        for d in (64,128):
            kh=2;r=16;nf=37
            for t in (1,4,17,129,1025,4097):
                raw=torch.randn(t,kh*d,generator=rng)*.3
                if t in (1,4):raw.zero_()
                packed=quantize_nvfp4(raw)
                model=dict(center=torch.zeros(kh,d),scale=torch.ones(kh,d),index=torch.arange(r)[None].expand(kh,-1).contiguous(),
                    feature_scale=torch.ones(kh,nf),beta=torch.randn(kh,nf,2+d,generator=rng)*.05,
                    active=torch.tensor([True,False]),uncertainty=True,gradient_gain=m._AM_GRADIENT_GAIN,parent=None)
                state=dict(moment=model);saved=copy.deepcopy((packed,state));random_state=torch.random.get_rng_state().clone()
                if args.factor:
                    model=dict(center=torch.zeros(kh,d),scale=torch.ones(kh,d),
                        beta=torch.randn(kh,d+1,16,generator=rng)*.05,search_ridge=1.,
                        active=torch.tensor([True,False]),parent=None)
                    state=dict(moment=model);saved=copy.deepcopy((packed,state))
                assert not validate_attention_calibration_result(dict(q_state=None,k_state=None,v_state=state),'attention')
                out=m.hif4_dynamic_quantize_v(*packed,kh,d,state)
                assert not validate_hif4_params(out,packed[0].shape,'v')
                assert tree_equal(out,other.hif4_dynamic_quantize_v(*packed,kh,d,copy.deepcopy(state)))
                base=m._am_parent_v(*packed,kh,d,None)
                assert all(torch.equal(p[:,d//64:],base[k][:,d//64:]) for k,p in out.items())
                if t in (1,4097):assert tree_equal(base,out)
                assert tree_equal(saved,(packed,state)) and torch.equal(random_state,torch.random.get_rng_state())
                a,var,g=m._am_moments(m.dequantize_nvfp4(*packed).float(),model,kh,d)
                assert torch.isfinite(a).all() and torch.isfinite(var).all() and torch.isfinite(g).all()
                assert torch.allclose(a.sum(1),torch.ones(kh)) and bool((var>=0).all())
                checks+=7
        with tempfile.TemporaryDirectory(prefix=f'hif4-v{version}-') as folder:
            (ROOT.__class__(folder)/'solution.py').write_text(source)
            result=subprocess.run([sys.executable,str(ROOT/'self_check.py'),'--solution_dir',folder,'--datasets_dir',str(ROOT/'mini_sample')],
                cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),capture_output=True,text=True,timeout=240)
        report['versions'][version]=dict(checks=checks,official_passed=result.returncode==0,stdout=result.stdout,stderr=result.stderr)
        print(version,checks,result.returncode,flush=True);assert result.returncode==0,result.stdout+result.stderr
    report['complete']=True
    name='v351-self-check.json' if args.cal_cache else ('v350-self-check.json' if args.factor else 'v341-v342-self-check.json')
    (ROOT/'docs'/name).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
