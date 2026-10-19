"""Actual saved-wrapper dispatch, exception cleanup and unchanged definitions."""
import ast
import copy
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
torch.set_num_threads(4)
source=(ROOT/'solution/solution_v390.py').read_text()
parent=(ROOT/'solution/solution_v387.py').read_text()
assert source.startswith(parent)
tree=ast.parse(source)
allowed={'__future__','torch','numpy','typing','threading'}
for node in ast.walk(tree):
    if isinstance(node,ast.Import):assert all(a.name.split('.')[0] in allowed for a in node.names)
    if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] in allowed
suffix=ast.parse(source[len(parent):])
assert {n.name for n in suffix.body if isinstance(n,ast.FunctionDef)} == {'_fo_keys','_fo_prepare','hif4_calibration_attention'}

for failure in (None,'KT','TS','TEMP-fit','TEMP-guard'):
    m=load_solution(ROOT/'solution/solution_v390.py')
    trace=[]
    state={'q_state':{'gq':1},'k_state':{'gq':1},'v_state':{}}
    def base(*args):
        trace.append('BASE');return copy.deepcopy(state)
    def stage(name):
        def run(samples,st,*args):
            trace.append(name)
            if failure==name:raise RuntimeError('injected')
            return dict(st,k_state=dict(st['k_state'],**{name:True}),_ts_cache='local',_vr_cache='local')
        return run
    def fit(samples,st,*args):
        trace.append('TEMP-fit')
        if failure=='TEMP-fit':raise RuntimeError('injected')
        return None,None
    def guard(sample,st,delta,*args):
        trace.append('TEMP-guard')
        if failure=='TEMP-guard':raise RuntimeError('injected')
        return dict(st,q_state=dict(st['q_state'],temperature_gain=torch.ones(1)*1.2))
    def metric(samples,st,*args):
        trace.append('OJ-fit')
        assert ('temperature_gain' in st['q_state']) == (failure not in ('TEMP-fit','TEMP-guard'))
        assert ('KT' in st['k_state']) == (failure!='KT')
        assert ('TS' in st['k_state']) == (failure!='TS')
        assert '_ts_cache' not in st and '_vr_cache' not in st
        return None
    m._fo_base_calibration=base;m._kt_guard=stage('KT');m._ts_guard=stage('TS')
    m._temp_fit=fit;m._temp_guard=guard;m._oj_fit=metric
    sample={r:(torch.zeros(8,64),torch.ones(8,4)) for r in 'qkv'}
    result=m.hif4_calibration_attention([sample]*5,1,1,64)
    expected=['BASE','KT','TS','TEMP-fit']
    if failure!='TEMP-fit':expected+=['TEMP-guard']
    expected+=['OJ-fit']
    assert trace==expected,(failure,trace)
    assert m.FO_DEBUG['order']==['KT','TS','TEMP','OJ']
    assert '_ts_cache' not in result and '_vr_cache' not in result
    assert state=={'q_state':{'gq':1},'k_state':{'gq':1},'v_state':{}}
print('PASS: actual saved-wrapper order, each stage once, final K/TEMP seen by OJ, injected stage failures, cache cleanup, parent immutability, full source prefix, no dynamic/Linear redefinitions')
