import sys,copy
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
torch.set_num_threads(4)
m=load_solution(ROOT/'solution/solution_v380.py')
torch.manual_seed(780)
for d in (64,128,256):
    eye=torch.eye(d).expand(2,-1,-1).clone()
    rotation=m._lr_cayley(eye,torch.randn_like(eye),.03)
    rotation[1]=eye[1]
    torch.testing.assert_close(rotation.transpose(1,2)@rotation,eye,atol=1.e-6,rtol=1.e-6)
    h=torch.randn(2,d,d);h=h@h.transpose(1,2)/d+eye*.2
    st={'kvh':2,'rot':1,'u':m._upper_cholesky_inv(h),'H':h,'kt_probe':torch.randn(2,16,d)}
    st['ts_fallback']={k:v for k,v in st.items()}
    source=copy.deepcopy(st);changed=m._cr_role(st,rotation,2,d)
    assert m._cr_role(st,eye,2,d) is st
    assert changed['kt_probe'] is st['kt_probe']
    for side in (changed,changed['ts_fallback']):
        u=st['u'];expected=rotation.transpose(1,2)@(u.transpose(1,2)@u)@rotation
        torch.testing.assert_close(side['u'].transpose(1,2)@side['u'],expected,atol=2.e-5,rtol=2.e-5)
        assert torch.equal(side['u'][1],st['u'][1])
        assert torch.equal(side['H'][1],st['H'][1])
    q=torch.randn(7,4*d);k=torch.randn(19,2*d)
    q0=m._cov_transform(q,st,4,2,d,'q');k0=m._cov_transform(k,st,2,2,d,'k').repeat_interleave(2,0)
    q1=m._cov_transform(q,changed,4,2,d,'q');k1=m._cov_transform(k,changed,2,2,d,'k').repeat_interleave(2,0)
    torch.testing.assert_close(q0@k0.transpose(1,2),q1@k1.transpose(1,2),atol=4.e-5,rtol=4.e-5)
    assert torch.equal(st['u'],source['u'])
print('PASS: orthogonality, GQA logits, recursive inverse metric, raw probes and inactive heads')
