"""Bounded calibration-only exact K cache; never retained by a test interface."""
_AC_ACTIVE=None
_AC_LAST_STATS={}
_ac_parent_calibration=hif4_calibration_attention
_ac_parent_key=hif4_dynamic_quantize_k


def _ac_bytes(x):
    if isinstance(x,torch.Tensor):return x.numel()*x.element_size()
    if isinstance(x,dict):return sum(_ac_bytes(v) for v in x.values())
    if isinstance(x,(tuple,list)):return sum(_ac_bytes(v) for v in x)
    if x is None or isinstance(x,(bool,int,float,str)):return 0
    raise TypeError('unsupported cache value')


def _ac_copy(x):
    if isinstance(x,torch.Tensor):return x.detach().clone()
    if isinstance(x,dict):return {k:_ac_copy(v) for k,v in x.items()}
    if isinstance(x,list):return [_ac_copy(v) for v in x]
    if isinstance(x,tuple):return tuple(_ac_copy(v) for v in x)
    return x


def _ac_equal(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,torch.Tensor):return a.dtype==b.dtype and a.device==b.device and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(_ac_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(_ac_equal(x,y) for x,y in zip(a,b))
    return a==b


def hif4_dynamic_quantize_k(k_quant,k_scale,kv_num_heads,head_dim,k_state):
    cache=_AC_ACTIVE
    if cache is None or not any(q is k_quant and s is k_scale for q,s in cache['targets']):
        return _ac_parent_key(k_quant,k_scale,kv_num_heads,head_dim,k_state)
    cache['calls']+=1
    for entry in reversed(cache['entries']):
        if entry['q'] is k_quant and entry['s'] is k_scale and entry['dims']==(kv_num_heads,head_dim):
            if torch.equal(entry['qs'],k_quant) and torch.equal(entry['ss'],k_scale) and _ac_equal(entry['state'],k_state):
                cache['hits']+=1
                return _ac_copy(entry['out'])
    out=_ac_parent_key(k_quant,k_scale,kv_num_heads,head_dim,k_state)
    size=_ac_bytes((k_quant,k_scale,k_state,out))
    if size<=67108864:
        while cache['entries'] and (len(cache['entries'])>=6 or cache['bytes']+size>67108864):
            cache['bytes']-=cache['entries'].pop(0)['bytes']
        cache['entries'].append(dict(q=k_quant,s=k_scale,qs=k_quant.detach().clone(),ss=k_scale.detach().clone(),
            dims=(kv_num_heads,head_dim),state=_ac_copy(k_state),out=_ac_copy(out),bytes=size))
        cache['bytes']+=size;cache['peak_bytes']=max(cache['peak_bytes'],cache['bytes'])
    return out


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    global _AC_ACTIVE
    # Do not share a cache with a nested calibration invocation.
    pool=[s for s in calib_qkv_list[:-1] if 4<=s['v'][0].shape[0]<=2048 and s['v'][0].numel()<=4194304][-2:]
    if calib_qkv_list:pool=pool+[calib_qkv_list[-1]]
    cache=dict(targets=[s['k'] for s in pool],entries=[],bytes=0,peak_bytes=0,hits=0,calls=0)
    _AC_ACTIVE=cache
    try:return _ac_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    finally:
        _AC_LAST_STATS.clear();_AC_LAST_STATS.update(hits=cache['hits'],calls=cache['calls'],peak_bytes=cache['peak_bytes'])
        cache['entries'].clear();cache['targets'].clear()
        _AC_ACTIVE=None
