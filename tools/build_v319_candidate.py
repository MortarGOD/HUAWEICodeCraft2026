"""Static source construction, never included as a submission dependency."""
import ast
from pathlib import Path
from quant_three_levels import rewrite
ROOT=Path(__file__).resolve().parents[1]


def candidate_source(reuse=True, route=True, neighbors=True, rounds=4):
    source=(ROOT/'solution/solution_v318.py').read_text()
    if reuse:
        source=rewrite(source)
    core=(ROOT/'tools/value_metadata_core.py').read_text()
    node=next(n for n in ast.parse(core).body if isinstance(n,ast.FunctionDef) and n.name=='_vm_proposal')
    core=core.replace(ast.get_source_segment(core,node),'')
    inner=(ROOT/'tools/value_metadata_inner_core.py').read_text()
    node=next(n for n in ast.parse(inner).body if isinstance(n,ast.FunctionDef) and n.name=='_vm_refine_new_tree')
    core+='\n_VM_INNER_ROUNDS='+str(int(rounds))+'\n'+ast.get_source_segment(inner,node)
    core+='\n\n'+(ROOT/'tools/value_metadata_row_core.py').read_text()
    # Fixed configuration; no nested fallback wrappers from other candidates.
    core=core.replace('_VM_VARIANCE=1.0','_VM_VARIANCE=8.0').replace('_VM_CAP=1.01','_VM_CAP=1.1')
    node=next(n for n in ast.parse(core).body if isinstance(n,ast.FunctionDef) and n.name=='hif4_dynamic_quantize_v')
    old=ast.get_source_segment(core,node)
    new='''def hif4_dynamic_quantize_v(q,s,kh,d,state):
    if not isinstance(state,dict) or 'vm_active' not in state:
        return _vm_parent_dynamic(q,s,kh,d,state)
    parent=_vm_parent_dynamic(q,s,kh,d,state['fallback'])
    heads=state['vm_active']
    rows=q.shape[0]
    if rows<=1 or rows>4096 or not bool(heads.any()):
        return parent
    # Value scale/mean objectives are column independent. Inactive KV heads
    # need no candidate trees or feedback iterations and stay byte-identical.
    active=heads.repeat_interleave(d//64)
    qa=q.reshape(rows,kh,d)[:,heads].reshape(rows,-1).contiguous()
    sa=s.reshape(rows,kh,d//16)[:,heads].reshape(rows,-1).contiguous()
    subset={key:value[:,active].contiguous() for key,value in parent.items()}
    proposal=_vm_proposal(dequantize_nvfp4(qa,sa).float(),subset,state['vm_variance'])
    result={key:value.clone() for key,value in parent.items()}
    for key in result:
        result[key][:,active]=proposal[key]
    return result'''
    core=core.replace(old,new)
    if neighbors:
        core=core.replace('    for specification in CAND_GRID:\n        candidate=_vm_refine_new_tree(x,_quantize_weighted(x,torch.ones(1,channels),grid=(specification,)),variance)',
            '    for sf in _vm_neighbor_scales(parent["scale_factor"]):\n        candidate=_vm_refine_new_tree(x,_vm_tree_at_sf(x,sf),variance)')
        core+='''

def _vm_neighbor_scales(sf):
    power=torch.exp2(torch.floor(torch.log2(sf)))
    # Adjacent E6M2 values; the lower step halves at exponent boundaries.
    lower=sf-torch.where(sf==power,power*.125,power*.25)
    upper=sf+power*.25
    return (lower.clamp(SF_MIN,SF_MAX),sf,upper.clamp(SF_MIN,SF_MAX))


def _vm_tree_at_sf(x,sf):
    xb=x.unflatten(-1,(-1,8,2,4))
    ab=xb.abs()
    errors=[]
    for factor in (1.0,2.0,4.0):
        unit=sf*factor
        value=(ab/unit*4).round().clamp(0,7)*.25*unit
        errors.append((value-ab).square().sum(-1))
    best=None
    for index,lv2 in enumerate((1.0,2.0)):
        first=errors[index]<=errors[index+1]
        local=torch.where(first,errors[index],errors[index+1])
        lv3=torch.where(first,1.0,2.0)
        total=local.sum(-1)
        if best is None:
            best=total;l2=torch.ones_like(total);l3=lv3
        else:
            take=total<best
            best=torch.minimum(best,total)
            l2=torch.where(take,lv2,l2)
            l3=torch.where(take[...,None],lv3,l3)
    l2=l2[...,None,None];l3=l3[...,None]
    unit=sf*l2*l3
    return dict(scale_factor=sf,scale_lv2=l2,scale_lv3=l3,sign=xb.sign(),
                mant=(ab/unit*4).round().clamp(0,7)*.25)
'''
    if route:
        core=core.replace("        raw=[dequantize_nvfp4(*hold[r]).float() for r in 'qkv']", """        eligible=_vm_eligible(hold,qh,kh,d)
        VM_DEBUG['eligible_heads']=int(eligible.sum())
        if not bool(eligible.any()):
            return states
        raw=[dequantize_nvfp4(*hold[r]).float() for r in 'qkv']""")
        core=core.replace('        proposal=_vm_proposal(raw[2],parent,_VM_VARIANCE)', """        blocks=eligible.repeat_interleave(d//64)
        subset={key:value[:,blocks].contiguous() for key,value in parent.items()}
        selected=raw[2].reshape(rows,kh,d)[:,eligible].reshape(rows,-1).contiguous()
        trial=_vm_proposal(selected,subset,_VM_VARIANCE)
        proposal={key:value.clone() for key,value in parent.items()}
        for key in proposal:
            proposal[key][:,blocks]=trial[key]""")
        core=core.replace('        mask=(costs[1]<costs[0]*.995).all(0)', '        mask=(costs[1]<costs[0]*.995).all(0)&eligible')
        core+='''

def _vm_eligible(hold,qh,kh,d):
    # The Value mean/variance surrogate is intended for diffuse Attention.
    # Estimate inverse participation using calibration queries and EVERY key.
    # A factor <=16 means at least T/16 effective keys, on average.
    rows=hold['q'][0].shape[0]
    ix=torch.linspace(0,rows-1,min(rows,16)).long()
    q=dequantize_nvfp4(hold['q'][0][ix],hold['q'][1][ix]).float()
    k=dequantize_nvfp4(*hold['k']).float()
    q=q.reshape(-1,qh,d).transpose(0,1)
    k=k.reshape(rows,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
    probability=torch.softmax(q@k.transpose(1,2)/(d**.5),-1)
    concentration=rows*probability.square().sum(-1).mean(-1).reshape(kh,qh//kh).mean(-1)
    return torch.isfinite(concentration)&(concentration<=16.0)
'''
    # Preserve the official public keyword names as well as positional calls.
    core=core.replace('def hif4_calibration_attention(samples,qh,kh,d):',
        'def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):\n    samples,qh,kh,d=calib_qkv_list,q_num_heads,kv_num_heads,head_dim')
    core=core.replace('def hif4_dynamic_quantize_v(q,s,kh,d,state):',
        'def hif4_dynamic_quantize_v(v_quant,v_scale,kv_num_heads,head_dim,v_state):\n    q,s,kh,d,state=v_quant,v_scale,kv_num_heads,head_dim,v_state')
    # Remove historical research labels from the actual submission header.
    end=source.index('"""',source.index('"""')+3)+3
    return '''"""v319: guarded Value scale-tree/rounding co-optimization.

Linear and Q/K algorithms frozen at v318. Value candidates stay on legal
HiF4 grids, constrain parent MSE, and are selected on calibration output.
"""'''+source[end:]+'\n\n'+core
