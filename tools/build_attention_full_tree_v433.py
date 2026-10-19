"""Build a bounded full-format, constrained single-token V coordinate solver."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'solution/solution_v431.py').read_text()
prefix=source[:source.index('\ndef _a431_tree')]
suffix=source[source.index('\ndef hif4_dynamic_quantize_v(vq,vs,kh,d,state):'):]
suffix=suffix.replace('_a431_tree','_a433_tree').replace('a431_active','a433_active')
body='''

def _a433_solve_nodes(optimum,lower,upper):
    # Exhaustive SF and exact nested minima: lv3 per four values, lv2 per eight.
    sf=_decode_e6m2(torch.arange(255))
    o=optimum.reshape(-1,1,8,2,4)
    low=lower.reshape_as(o);high=upper.reshape_as(o)
    best8=None;best_codes=None;best_lv3=None;best_lv2=None
    for lv2 in (1.,2.):
        best4=None;codes4=None;level3=None
        for lv3 in (1.,2.):
            step=sf.reshape(1,-1,1,1,1)*(lv2*lv3*.25)
            lo=torch.ceil(low/step).clamp_min(-7.)
            hi=torch.floor(high/step).clamp_max(7.)
            code=torch.minimum(torch.maximum((o/step).round().clamp(-7,7),lo),hi)
            cost=(code*step-o).square().sum(-1)
            cost=cost.masked_fill((lo>hi).any(-1),torch.inf)
            if best4 is None:
                best4=cost;codes4=code;level3=torch.full_like(cost,lv3)
            else:
                take=cost<best4
                best4=torch.minimum(best4,cost)
                codes4=torch.where(take[:,:,:,:,None],code,codes4)
                level3=torch.where(take,torch.full_like(cost,lv3),level3)
        cost8=best4.sum(-1)
        if best8 is None:
            best8=cost8;best_codes=codes4;best_lv3=level3
            best_lv2=torch.full_like(cost8,lv2)
        else:
            take=cost8<best8
            best8=torch.minimum(best8,cost8)
            best_codes=torch.where(take[:,:,:,None,None],codes4,best_codes)
            best_lv3=torch.where(take[:,:,:,None],level3,best_lv3)
            best_lv2=torch.where(take,torch.full_like(cost8,lv2),best_lv2)
    costs,index=best8.sum(-1).min(1)
    rows=torch.arange(optimum.shape[0])
    return sf[index],best_lv2[rows,index],best_lv3[rows,index],best_codes[rows,index],costs


def _a433_tree(x,params,state,kh,d):
    t=x.shape[0];blocks=d//64
    if not 2<=t<=4096 or d%64 or kh*blocks>64 or t*kh*d>2**23:return params
    probability=_ac_parent_probability(x,state,kh)
    if probability.shape[1]>128 or not bool(torch.isfinite(probability).all()):return params
    values=_deq_params(params).float().reshape(t,kh,d).transpose(0,1).contiguous()
    target=x.reshape(t,kh,d).transpose(0,1).contiguous()
    error=values-target
    gradient=probability.transpose(1,2)@(probability@error)+error/t
    diagonal=probability.square().sum(1)+1./t
    free_gain=(gradient.square()/diagonal[:,:,None]).reshape(kh,t,blocks,64).sum(-1)
    token=free_gain.argmax(1)
    hh=torch.arange(kh)[:,None].expand(kh,blocks)
    bb=torch.arange(blocks)[None].expand(kh,blocks)
    def selected(a):return a.reshape(kh,t,blocks,64)[hh,token,bb].reshape(-1,64)
    old=selected(values);raw=selected(target);g=selected(gradient);e=old-raw
    a=diagonal[hh,token].reshape(-1,1)
    cap=error.square().mean(1).clamp_min(1e-30).reshape(-1,64)
    radius=(e.square()+.05*t*cap).sqrt()
    optimum=old-g/a
    sf_parts=[];lv2_parts=[];lv3_parts=[];code_parts=[];cost_parts=[]
    for start in range(0,old.shape[0],16):
        sl=slice(start,start+16)
        result=_a433_solve_nodes(optimum[sl],(raw-radius)[sl],(raw+radius)[sl])
        for output,value in zip((sf_parts,lv2_parts,lv3_parts,code_parts,cost_parts),result):output.append(value)
    sf,lv2,lv3,codes,cost=[torch.cat(parts) for parts in (sf_parts,lv2_parts,lv3_parts,code_parts,cost_parts)]
    proposed=(sf[:,None,None,None]*lv2[:,:,None,None]*lv3[:,:,:,None]*codes*.25).reshape(-1,64)
    delta=proposed-old
    change=(2*g*delta+a*delta.square()).sum(-1)
    new_mse=cap+(2*e*delta+delta.square())/t
    active=state['a433_active'][:,None].expand(kh,blocks).reshape(-1)
    take=active&torch.isfinite(cost)&torch.isfinite(change)&(change<0)&(new_mse<=cap*1.05).all(-1)
    if not bool(take.any()):return params
    positions=(token.reshape(-1)[take],hh.reshape(-1)[take],bb.reshape(-1)[take])
    result=dict(params)
    updates={'scale_factor':sf[:,None,None,None],'scale_lv2':lv2[:,:,None,None],
             'scale_lv3':lv3[:,:,:,None],'sign':codes.sign(),'mant':codes.abs()*.25}
    for key,value in updates.items():
        result[key]=params[key].clone()
        shape=value.shape[1:]
        result[key].reshape(t,kh,blocks,*shape)[positions]=value[take].to(result[key].dtype)
    return result
'''
text=prefix+body+suffix
ast.parse(text)
(ROOT/'solution/solution_v433.py').write_text(text)
print('Built v433; full 255-SF constrained tree at one screened token per 64 channels.')
