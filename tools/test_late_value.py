import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
m=load_solution(ROOT/'solution/solution_v384.py')
assert m._oj_parent_calibration is m._vr_parent_calibration
assert (ROOT/'solution/solution_v384.py').read_text().startswith((ROOT/'solution/solution_v382.py').read_text())
events=[];parent={'q_state':{},'k_state':{},'v_state':{}}
def prior(*args):events.append('VM-OJ-KT-TS-TEMP');return parent
def guard(samples,state,*args):
    events.append('VR');assert state is parent
    return dict(state,_vr_cache=['temporary'],_ts_cache=['temporary'])
m._lv_parent_calibration=prior;m._vr_guard=guard
r=m.hif4_calibration_attention([{},{}],2,1,64)
assert events==['VM-OJ-KT-TS-TEMP','VR']
assert '_vr_cache' not in r and '_ts_cache' not in r
events.clear()
def fail(*args):events.append('VR');raise RuntimeError('diagnostic')
m._vr_guard=fail
r=m.hif4_calibration_attention([{},{}],2,1,64)
assert r==parent and m.LV_DEBUG['error']
assert events==['VM-OJ-KT-TS-TEMP','VR']
print('PASS: VM retained, late VR once, cache removal and exception fallback')
