import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
m=load_solution(ROOT/'solution/solution_v383.py')
for gain,want in [(None,False),(torch.ones(8),False),(torch.tensor([1.,1.000001]),True),(torch.tensor([.99,1.]),True)]:
    state={'q_state':{} if gain is None else {'temperature_gain':gain}}
    assert m._sg_protected(state)==want
parent={'q_state':{'gq':1,'temperature_gain':torch.tensor([1.,1.01])},'k_state':{},'v_state':{}}
m._sg_parent_calibration=lambda *a:parent
def forbidden(*args):raise AssertionError('protected route must not fit/guard')
m._temp_fit=forbidden;m._temp_guard=forbidden
assert m.hif4_calibration_attention([],2,1,64) is parent
assert m.SG_DEBUG['protected']
parent['q_state']['temperature_gain']=torch.ones(2);calls=[]
samples=[{'v':(torch.ones(4,64),None)}]*2
def fit(*args):calls.append('fit');return torch.zeros(2),[]
def guard(*args):calls.append('guard');return parent
m._temp_fit=fit;m._temp_guard=guard
r=m.hif4_calibration_attention(samples,2,1,64)
assert calls==['fit','guard'] and not m.SG_DEBUG['protected']
assert r['q_state'] is parent['q_state']
print('PASS: calibration-only predicate, protected exact-state bypass, unit-gain late fit')
