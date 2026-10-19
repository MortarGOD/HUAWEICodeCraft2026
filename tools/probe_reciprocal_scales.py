"""Compare zero-step initialization against learned reciprocal scale candidates."""
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
from probe_attention_gates import configured
from reciprocal_scale_core import train
from evaluate_local import evaluate_attention
from self_check import validate_frozen_state,validate_hif4_params


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups',default='5,14,19')
    p.add_argument('--steps',type=int,default=4)
    p.add_argument('--rate',type=float,default=.05)
    p.add_argument('--queries',type=int,default=32)
    p.add_argument('--output',default='docs/reciprocal-scale-screen.json')
    a=p.parse_args();bank=torch.load(ROOT/a.dataset/'attn.pt',weights_only=False);groups=[bank[int(i)] for i in a.groups.split(',')]
    m=configured('frames',0);parents={};extras={};training=[]
    for gi,g in zip(map(int,a.groups.split(',')),groups):
        qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        parent=m._gate_parent_calibration(g['calib'],qh,kh,d);parents[id(g['calib'])]=parent
        extras[id(g['calib'])]=[]
        for plain in (False,True):
            initial,learned,debug=train(m,g['calib'],parent,qh,kh,d,plain,a.steps,a.rate,a.queries)
            training.append(dict(group=gi,**debug))
            if initial is not None:
                assert not validate_frozen_state(initial,'scale_initial')
                assert not validate_frozen_state(learned,'scale_trained')
                extras[id(g['calib'])].append((initial,learned,'plain' if plain else 'parent_basis'))
            print('trained',gi,plain,debug,flush=True)
    report=dict(dataset=a.dataset,groups=a.groups,steps=a.steps,rate=a.rate,queries=a.queries,training=training,
                paired_parent=True,results={},warning='Actual quantized forward, approximate STE direction. Initial-state controls isolate learning; no online forecast.')
    for mode in ('none','frames','initial','learned'):
        module=configured('frames',0);original=module._gate_frame_candidates
        def cached(samples,*args):return copy.deepcopy(parents[id(samples)])
        module._gate_parent_calibration=cached
        if mode=='none':module.hif4_calibration_attention=cached
        if mode in ('initial','learned'):
            def candidates(samples,parent,qh,kh,d,rows):
                frames,labels=original(samples,parent,qh,kh,d,rows)
                for initial,learned,label in extras[id(samples)]:
                    frames.append(initial);labels.append('init_'+label)
                    if mode=='learned':frames.append(learned);labels.append('learn_'+label)
                return frames,labels
            module._gate_frame_candidates=candidates
        cal=module.hif4_calibration_attention;diags=[];checks=dict(states=0,returns=0)
        for role in 'qkv':
            original_dynamic=getattr(module,'hif4_dynamic_quantize_'+role)
            def checked(*args,_fn=original_dynamic,_role=role):
                out=_fn(*args);errors=validate_hif4_params(out,args[0].shape,'scale_'+_role);assert not errors,errors
                checks['returns']+=1;return out
            setattr(module,'hif4_dynamic_quantize_'+role,checked)
        def capture(*args):
            st=cal(*args);errors=validate_frozen_state(st,'scale_state');assert not errors,errors
            checks['states']+=1;diags.append(dict(module.GATE_DEBUG));return st
        module.hif4_calibration_attention=capture
        with contextlib.redirect_stdout(io.StringIO()):scores,seconds=evaluate_attention(module,groups)
        report['results'][mode]=dict(score=sum(scores),cases=scores,seconds=seconds,diagnostics=diags,checks=checks)
        (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')
        print(mode,sum(scores),[(x.get('frame_choice'),x.get('frame_labels')) for x in diags],flush=True)


if __name__=='__main__':main()
