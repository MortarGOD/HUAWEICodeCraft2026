import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
torch.set_num_threads(4)
m=load_solution(ROOT/'solution/solution_v386.py')
torch.manual_seed(386)
for d in (64,128,256):
    eye=torch.eye(d).expand(2,-1,-1).clone()
    old={'H':eye,'u':eye,'kvh':2,'gq':1}
    x=torch.randn(2,d,8);h=m._oj_normalize(x@x.transpose(1,2))
    st=m._oj_state(old,h)
    torch.testing.assert_close(st['H'],h)
    damp=h.diagonal(dim1=-2,dim2=-1).mean(-1)*m.GPTQ_DAMP
    inverse=st['u'].transpose(1,2)@st['u']
    torch.testing.assert_close(inverse@(h+damp[:,None,None]*eye),eye,atol=2.e-3,rtol=2.e-3)
    for bad in (torch.zeros_like(h),torch.full_like(h,float('nan'))):
        fallback=m._oj_state(old,bad)
        assert fallback['H'] is old['H'] and fallback['u'] is old['u']
    nested=dict(old,ts_fallback=old)
    result=m._oj_state(nested,h)
    torch.testing.assert_close(result['ts_fallback']['H'],h)
    assert torch.equal(old['H'],eye)
print('PASS: pure metric, rank-deficient damping, zero/NaN fallback, recursive state, immutable parent')
