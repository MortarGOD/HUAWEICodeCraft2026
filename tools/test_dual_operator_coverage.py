"""D512 implementation and D256 unchanged; gate-only tests are not gain evidence."""
import ast,copy,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from self_check import validate_hif4_params,validate_frozen_state
torch.set_num_threads(4);torch.manual_seed(406)
old=load_solution(ROOT/'solution/solution_v405.py');new=load_solution(ROOT/'solution/solution_v406.py')
source=(ROOT/'solution/solution_v406.py').read_text()
assert source.startswith((ROOT/'solution/solution_v405.py').read_text())
for node in ast.walk(ast.parse(source)):
    if isinstance(node,ast.Import):assert all(a.name.split('.')[0] in {'torch','numpy','typing','threading'} for a in node.names)
    if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] in {'__future__','typing','torch','numpy','threading'}
for d in (256,512):
    n=129;kh=1;qh=8;x=torch.randn(n,d)
    st=dict(center=torch.zeros(d),projection=torch.randn(d,4)*.03,raw_kernel=torch.randn(qh,5,5)*.1,
            modes=torch.randn(qh,2,5)*.1,active=torch.ones(kh,dtype=torch.bool))
    st['quant_kernel']=st['raw_kernel']+torch.randn(qh,5,5)*.03
    snapshot=copy.deepcopy(st);rng=torch.random.get_rng_state().clone()
    p=new._dyn_table(x,None,False);a=new._do_refine(x,p,st,kh,d)
    assert not validate_hif4_params(a,x.shape,'width')
    assert not validate_frozen_state(st,'width')
    raw,quant=new._do_probabilities(x,st,kh);r=quant.shape[1]
    ridge=r/n**2+.05*quant.square().sum(2).mean(1)
    xh=x[None]
    def loss(p):
        y=new._deq_params(p)[None];mse=(xh-y).square().mean(1)
        return (raw@xh-quant@y).square().mean(1)+(ridge*n/r)[:,None]*mse,mse
    c0,e0=loss(p);c1,e1=loss(a)
    assert bool((c1<=c0+1e-7).all()) and bool((e1<=e0*1.5+1e-7).all())
    assert all(torch.equal(v,st[k]) for k,v in snapshot.items()) and torch.equal(rng,torch.random.get_rng_state())
    if d==256:
        b=old._do_refine(x,p,st,kh,d)
        assert all(torch.equal(v,b[k]) for k,v in a.items())
    print('PASS width',d,'risk rows',r,'changed',int((new._deq_params(a)!=new._deq_params(p)).sum()))
# Check only the shape routing here, then use real dataset runs for activation.
parent=dict(q_state={'gq':1},k_state={'gq':1},v_state=None)
samples=[{'v':(torch.zeros(8,512),None)}]*3
for module,expected in ((old,0),(new,1)):
    calls=[0]
    def fit(*a):calls[0]+=1;return None
    module._do_fit=fit
    assert module._do_guard(samples,parent,8,1,512) is parent
    assert calls[0]==expected
calls[0]=0
assert new._do_guard(samples,parent,8,1,1024) is parent and calls[0]==0
print('PASS parent prefix/imports, D256 exact, D512 real solver objective/MSE/format/state, D512 routing and D1024 bypass')
