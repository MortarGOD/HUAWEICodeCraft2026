"""Build a static single-file weighted-origin candidate; not submitted."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def candidate_source():
    parent=(ROOT/'solution/solution_v321.py').read_text()
    # Invocation-local reuse only. Do not put decoded calibration workspaces
    # into the frozen state or use a global communication channel.
    parent=parent.replace('old=cost(k);best=old.clone();chosen=',
                          'old=cost(k);best=old.clone();best_key=k;chosen=')
    parent=parent.replace('best=torch.where(take[None],new,best);chosen=torch.where(take,mode,chosen)',
        'best_key=torch.where(take.repeat_interleave(d)[None],candidate,best_key)\n        best=torch.where(take[None],new,best);chosen=torch.where(take,mode,chosen)')
    parent=parent.replace('active=chosen>0\n    KT_DEBUG.update',
        "active=chosen>0\n    parent=dict(parent,_ts_cache=[ix,raw,target,q,best_key,v])\n    KT_DEBUG.update")
    parent=parent.replace('return dict(parent,k_state=candidate_state)',
        "if bool((chosen==2).any()) and bool((chosen==4).any()):\n        parent['_ts_cache'][4]=_deq_params(_kt_quant(raw[1],candidate_state,kh,d,mode))\n    return dict(parent,k_state=candidate_state)")
    core=(ROOT/'tools/attention_translation_search_core.py').read_text()
    tree=ast.parse(core)
    # Keep the selected algorithm, not experimental quantizer branches.
    quant='''def _ts_quant(raw,st,kh,d,variant):
    y=_kt_input(raw,st,kh,d,'k').reshape(-1,kh,d)
    key=raw.reshape(-1,kh,d).transpose(0,1)
    prob=torch.softmax(st['kt_probe'].float()@key.transpose(1,2),-1)
    weight=(.9*prob.mean(1)+.1/raw.shape[0]).transpose(0,1)[:,:,None]
    center=(y*weight).sum(0,keepdim=True)
    for _ in range(8):
        z=y-center
        error=_deq_params(_dyn_table(z.reshape(raw.shape),None,has_scale=False)).reshape(y.shape)-z
        center=center-(error*weight).sum(0,keepdim=True)
    plain={k:v for k,v in st.items() if k not in ('joint_transform','ks','rot','hc_matrix')}
    return _kt_fixed_grid((y-center).reshape(raw.shape).contiguous(),plain,kh,d,'k')
'''
    guard=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_ts_guard')
    guard.body=[n for n in guard.body if not (isinstance(n,ast.If) and ast.unparse(n.test)=='_TS_VARIANT in (10, 11)')]
    names=('hif4_calibration_attention','hif4_dynamic_quantize_k')
    functions=[ast.get_source_segment(core,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    tail='\n\nTS_DEBUG={}\n_TS_VARIANT=1\n_TS_MATCH_GUARD=True\n_ts_parent_calibration=hif4_calibration_attention\n_ts_parent_k=hif4_dynamic_quantize_k\n\n'
    end=parent.index('"""',parent.index('"""')+3)+3
    header='''"""v322: Attention-weighted Key origin with deployment-gated offset fitting.
Linear, Query and Value are unchanged from v321. No runtime file access.
"""'''
    source=header+parent[end:]+tail+quant+'\n\n'+ast.unparse(guard)+'\n\n'+'\n\n'.join(functions)+'\n'
    compile(source,'v322','exec')
    return source


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args()
    text=candidate_source()
    if args.write:
        for name in ('solution_v322.py','solution.py'):(ROOT/'solution'/name).write_text(text)
    print('lines',len(text.splitlines()))
