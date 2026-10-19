from pathlib import Path
import ast
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
torch.set_num_threads(4)
torch.manual_seed(388)
path=ROOT/'solution/solution_v388.py'
src=path.read_text()
assert src.startswith((ROOT/'solution/solution_v387.py').read_text())
for node in ast.walk(ast.parse(src)):
    names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) else []
    assert all(x.split('.')[0] in {'__future__','torch','numpy','typing','threading'} for x in names)
m=load_solution(path)
for d in (64,128,256):
    old={'H':torch.eye(d).repeat(2,1,1),'u':torch.eye(d).repeat(2,1,1),'kvh':2}
    x=torch.randn(8,d,8)
    h=m._oj_normalize(x@x.transpose(1,2))
    trial=m._oj_state(old,h)
    torch.testing.assert_close(trial['H'],.5*(torch.eye(d)[None]+h))
    assert trial['u'].shape==(8,d,d) and trial['kvh']==2
    result=m._oj_merge(old,trial,torch.tensor([False,True]))
    torch.testing.assert_close(result['H'][:4],old['H'][0].expand(4,-1,-1))
    torch.testing.assert_close(result['H'][4:],trial['H'][4:])
    torch.testing.assert_close(result['u'][:4],old['u'][0].expand(4,-1,-1))
    nested=dict(old,ts_fallback=old)
    ntrial=m._oj_state(nested,h)
    nresult=m._oj_merge(nested,ntrial,torch.tensor([True,False]))
    torch.testing.assert_close(nresult['ts_fallback']['H'][4:],old['H'][1].expand(4,-1,-1))
    assert old['H'].shape[0]==2

d=64; qh=8; kh=2; n=17
sample={r:(torch.zeros(n,qh*d if r=='q' else kh*d),torch.ones(1)) for r in 'qkv'}
q=torch.randn(16,qh*d); k=torch.randn(n,kh*d); v=torch.randn(n,kh*d)
m._joint_quant_queries=lambda *a:q
m.hif4_dynamic_quantize_k=lambda *a:k
m.hif4_dynamic_quantize_v=lambda *a:v
m._deq_params=lambda x:x
states={r+'_state':{} for r in 'qkv'}
fit=m._oj_fit([sample]*5,states,qh,kh,d)[0]
expected=torch.stack([m._dm_query_gram(q.reshape(16,qh,d)[:,j],k.reshape(n,kh,d)[:,j//4],v.reshape(n,kh,d)[:,j//4],d) for j in range(qh)])
torch.testing.assert_close(fit,m._oj_normalize(expected))
assert not torch.allclose(fit[0],fit[1]) and m.PH_DEBUG['folds']==2
assert m._oj_fit([sample],states,qh,kh,d) is None
sentinel=object()
m._ph_parent_fit=lambda *a:sentinel
assert m._oj_fit([sample]*5,states,kh,kh,d) is sentinel
print('PASS: prefix/imports, per-Q metrics, group masks, shrinkage, immutable/recursive fallback, MHA bypass')
