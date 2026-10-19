"""Exact bounded K/V reuse within one calibration and one thread only."""
from threading import local as _VCThreadLocal
_VC_LOCAL=_VCThreadLocal()
_VC_LAST_STATS={}
_vc_parent_calibration=hif4_calibration_attention
_vc_parent_k=hif4_dynamic_quantize_k
_vc_parent_v=hif4_dynamic_quantize_v

def _vc_bytes(x):
    if isinstance(x,torch.Tensor):return x.numel()*x.element_size()
    if isinstance(x,dict):return sum(_vc_bytes(v) for v in x.values())
    if isinstance(x,(tuple,list)):return sum(_vc_bytes(v) for v in x)
    if x is None or isinstance(x,(bool,int,float,str)):return 0
    raise TypeError('unsupported cache value')


def _vc_copy(x):
    if isinstance(x,torch.Tensor):return x.detach().clone()
    if isinstance(x,dict):return {k:_vc_copy(v) for k,v in x.items()}
    if isinstance(x,list):return [_vc_copy(v) for v in x]
    if isinstance(x,tuple):return tuple(_vc_copy(v) for v in x)
    return x


def _vc_equal(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,torch.Tensor):return a.dtype==b.dtype and a.device==b.device and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(_vc_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(_vc_equal(x,y) for x,y in zip(a,b))
    return a==b



def _vc_call(role,fn,q,s,h,d,state):
    cache=getattr(_VC_LOCAL,'active',None)
    if cache is None or not any(r==role and a is q and b is s for r,a,b in cache['targets']):
        return fn(q,s,h,d,state)
    cache['calls'][role]+=1
    for entry in reversed(cache['entries']):
        if entry['role']==role and entry['q'] is q and entry['s'] is s and entry['dims']==(h,d):
            if torch.equal(entry['qs'],q) and torch.equal(entry['ss'],s) and _vc_equal(entry['state'],state):
                cache['hits'][role]+=1
                return _vc_copy(entry['out'])
    out=fn(q,s,h,d,state)
    size=_vc_bytes((q,s,state,out))
    if size<=67108864:
        while cache['entries'] and (len(cache['entries'])>=8 or cache['bytes']+size>67108864):
            cache['bytes']-=cache['entries'].pop(0)['bytes']
        cache['entries'].append(dict(role=role,q=q,s=s,qs=q.detach().clone(),ss=s.detach().clone(),
            dims=(h,d),state=_vc_copy(state),out=_vc_copy(out),bytes=size))
        cache['bytes']+=size
        cache['peak_bytes']=max(cache['peak_bytes'],cache['bytes'])
    return out


def hif4_dynamic_quantize_k(k_quant,k_scale,kv_num_heads,head_dim,k_state):
    return _vc_call('k',_vc_parent_k,k_quant,k_scale,kv_num_heads,head_dim,k_state)


def hif4_dynamic_quantize_v(v_quant,v_scale,kv_num_heads,head_dim,v_state):
    return _vc_call('v',_vc_parent_v,v_quant,v_scale,kv_num_heads,head_dim,v_state)


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    previous=getattr(_VC_LOCAL,'active',None)
    pool=calib_qkv_list[-3:]
    cache=dict(targets=[(r,*s[r]) for s in pool for r in ('k','v')],
        entries=[],bytes=0,peak_bytes=0,hits={'k':0,'v':0},calls={'k':0,'v':0})
    _VC_LOCAL.active=cache
    try:
        return _vc_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    finally:
        _VC_LAST_STATS.clear()
        _VC_LAST_STATS.update(hits=dict(cache['hits']),calls=dict(cache['calls']),peak_bytes=cache['peak_bytes'])
        cache['entries'].clear();cache['targets'].clear()
        _VC_LOCAL.active=previous

