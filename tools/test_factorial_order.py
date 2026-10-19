import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
m=load_solution(ROOT/'solution/solution_v385.py')
assert m._lt_parent_calibration is m._co_metric_calibration
assert m._oj_parent_calibration is m._kf_before_metric
events=[];parent={'q_state':{'gq':1},'k_state':{'gq':0},'v_state':{}}
def early(*args):events.append('early');return parent
def kt(samples,state,*args):events.append('KT');return dict(state,_ts_cache=['temporary'])
def ts(samples,state,*args):events.append('TS');assert '_ts_cache' in state;return state
oj=m._lt_parent_calibration
def tracked_oj(*args):result=oj(*args);events.append('OJ');return result
def fit(samples,state,*args):events.append('TEMP-fit');assert '_ts_cache' not in state;return torch.zeros(1),[]
def guard(hold,state,*args):events.append('TEMP-guard');return state
m._kf_early_calibration=early;m._kt_guard=kt;m._ts_guard=ts
m._lt_parent_calibration=tracked_oj;m._temp_fit=fit;m._temp_guard=guard
samples=[{'v':(torch.ones(4,64),None)}]*2
result=m.hif4_calibration_attention(samples,1,1,64)
assert events==['early','KT','TS','OJ','TEMP-fit','TEMP-guard']
assert '_ts_cache' not in result and '_vr_cache' not in result
print('PASS: actual wrapper chain KT-TS-OJ-TEMP, each once, no temporary state')
