"""Order/fallback contract plus immutable dynamic-function source prefix."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
m=load_solution(ROOT/'solution/solution_v381.py')
assert (ROOT/'solution/solution_v381.py').read_text().startswith((ROOT/'solution/solution_v365.py').read_text())
events=[];original={'q_state':{'gq':0},'k_state':{'gq':0},'v_state':{}}
def early(*args):events.append('early');return original
def kt(samples,state,*args):
    events.append('KT');assert state is original
    return dict(state,_ts_cache=['temporary'])
def ts(samples,state,*args):
    events.append('TS');assert state['_ts_cache']==['temporary']
    return state
m._oj_parent_calibration=early;m._kt_guard=kt;m._ts_guard=ts
result=m.hif4_calibration_attention([],2,1,64)
assert events==['early','KT','TS'] and '_ts_cache' not in result
assert '_ts_cache' not in original
events.clear()
def broken(*args):events.append('TS');raise RuntimeError('diagnostic')
m._ts_guard=broken
result=m.hif4_calibration_attention([],2,1,64)
assert events==['early','KT','TS'] and '_ts_cache' not in result
assert m.CO_DEBUG['error'] and m.CO_DEBUG['stage']=='TS'
print('PASS: single early-parent/KT/TS calls, cache removal and guarded fallback')
