"""Guard reciprocal correction and scale-only removal with fixed parent V."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from probe_attention_gates import configured as gate_configured
from frame_causal_core import refit
from evaluate_local import evaluate_attention
from self_check import validate_frozen_state,validate_hif4_params


def configured(mode):
    m=gate_configured('frames',0)
    original=m._gate_frame_candidates
    if mode=='none':m.hif4_calibration_attention=m._gate_parent_calibration;return m
    if mode=='frames':return m
    def candidates(samples,parent,qh,kh,d,rows):
        bank,labels=original(samples,parent,qh,kh,d,rows) if mode=='combined' else ([parent],['parent'])
        if 'qs' in parent['q_state']:
            if mode in ('reciprocal','combined'):
                exact=dict(parent,k_state=dict(parent['k_state'],ks=parent['q_state']['qs'].float().reciprocal().contiguous()))
                bank.append(exact);labels.append('reciprocal_only')
            if mode in ('scale_only','combined') and 'joint_transform' in parent['q_state']:
                st=dict(parent)
                for role in 'qk':
                    field=dict(parent[role+'_state']);field.pop('qs' if role=='q' else 'ks',None)
                    st[role+'_state']=field
                bank.append(refit(m,samples,st,qh,kh,d));labels.append('no_scale_keep_dense')
        return bank,labels
    m._gate_frame_candidates=candidates
    return m


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups',default='5,13,14,19,23,47')
    p.add_argument('--modes',default='none,frames,reciprocal,combined')
    p.add_argument('--output',default='docs/frame-extended-screen.json')
    a=p.parse_args();bank=torch.load(ROOT/a.dataset/'attn.pt',weights_only=False);groups=[bank[int(i)] for i in a.groups.split(',')]
    report=dict(dataset=a.dataset,groups=a.groups,results={},warning='Calibration-selected candidates. Modeled inputs; no online score estimate.')
    # Pair all interventions to the exact same parent realization. Re-running
    # a numerically unstable parent factorization can confound small deltas.
    parent_module=gate_configured('frames',0);parents={};parent_debug=[]
    for g in groups:
        qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        parents[id(g['calib'])]=parent_module._gate_parent_calibration(g['calib'],qh,kh,d)
        parent_debug.append(dict(parent_module.JOINT_DEBUG))
    report['paired_parent']=True;report['parent_diagnostics']=parent_debug
    def cached_parent(samples,*args):return copy.deepcopy(parents[id(samples)])
    for mode in a.modes.split(','):
        m=configured(mode);m._gate_parent_calibration=cached_parent
        if mode=='none':m.hif4_calibration_attention=cached_parent
        cal=m.hif4_calibration_attention;diags=[];checks=dict(states=0,returns=0)
        for role in 'qkv':
            original=getattr(m,'hif4_dynamic_quantize_'+role)
            def checked(*args,_original=original,_role=role):
                out=_original(*args);errors=validate_hif4_params(out,args[0].shape,'extended_'+_role)
                assert not errors,errors
                checks['returns']+=1;return out
            setattr(m,'hif4_dynamic_quantize_'+role,checked)
        def capture(*args):
            st=cal(*args);errors=validate_frozen_state(st,'extended_state');assert not errors,errors
            checks['states']+=1;diags.append(dict(m.GATE_DEBUG));return st
        m.hif4_calibration_attention=capture
        with contextlib.redirect_stdout(io.StringIO()):scores,seconds=evaluate_attention(m,groups)
        report['results'][mode]=dict(score=sum(scores),cases=scores,seconds=seconds,diagnostics=diags,checks=checks)
        (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')
        print(mode,sum(scores),seconds,[(x.get('frame_choice'),x.get('frame_labels')) for x in diags],flush=True)


if __name__=='__main__':main()
