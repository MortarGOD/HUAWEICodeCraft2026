import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
m=load_solution(ROOT/'solution/solution_v382.py')
assert m._vm_parent_calibration is m._temp_parent_calibration
assert (ROOT/'solution/solution_v382.py').read_text().startswith((ROOT/'solution/solution_v381.py').read_text())
events=[];parent={'q_state':{'gq':1},'k_state':{},'v_state':{}}
samples=[{'v':(torch.ones(4,64),None)}]*2
def early(*args):events.append('V-OJ-KT-TS');return parent
def fit(samples,state,*args):events.append('fit');assert state is parent;return torch.ones(1),['local']
def guard(hold,state,delta,*args):events.append('guard');return dict(state,_vr_cache=['temporary'],_ts_cache=['temporary'])
m._lt_parent_calibration=early;m._temp_fit=fit;m._temp_guard=guard
r=m.hif4_calibration_attention(samples,1,1,64)
assert events==['V-OJ-KT-TS','fit','guard']
assert '_vr_cache' not in r and '_ts_cache' not in r
assert '_vr_cache' not in parent
print('PASS: early temperature bypass, one late fit/guard, clean state, parent prefix')
