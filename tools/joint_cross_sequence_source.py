"""Offline source construction for bounded auxiliary calibration evidence."""
HELPER='''
def _jc_aux_costs(hold, variants, kernel, qh, kh, d):
    rows=hold['v'][0].shape[0]
    ix=torch.linspace(0,rows-1,min(rows,32)).long()
    raw=[dequantize_nvfp4(hold['q'][0][ix],hold['q'][1][ix]).float()]
    raw.extend(dequantize_nvfp4(*hold[r]).float() for r in 'kv')
    vp0=_vk_parent_v(*hold['v'],None,kh,d)
    values=(_deq_params(vp0),_deq_params(_vk_refine(raw[2],vp0,kernel,kh,d)))
    target=_joint_output(*raw,qh,kh,d,ix,query_subset=True)
    costs=torch.empty(len(variants),2,2,kh)
    for bi,st in enumerate(variants):
        q=_joint_quant_queries(hold,st['q_state'],qh,kh,d,ix)
        k=_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,st['k_state']))
        for vi,v in enumerate(values):
            error=(_joint_output(q,k,v,qh,kh,d,ix,query_subset=True)-target).square().mean((-1,-2))
            costs[bi,vi]=torch.stack((error[::2].mean(0),error[1::2].mean(0)))
    return costs
'''


def candidate_source(source):
    needle='        bmask = torch.zeros(kh, dtype=torch.bool)\n        vmask = torch.zeros(kh, dtype=torch.bool)'
    assert source.count(needle)==1
    source=source.replace(needle,"""        auxiliary = None
        if len(calib_qkv_list)>=3 and 4<=calib_qkv_list[-2]['v'][0].shape[0]<=4096:
            auxiliary=_jc_aux_costs(calib_qkv_list[-2],variants,kernel,qh,kh,d)
            if not bool(torch.isfinite(auxiliary).all()):auxiliary=None
        rescues=0
"""+needle)
    source=source.replace('            best = costs[0, 0, :, hv]\n','            best = costs[0, 0, :, hv]\n            best_aux = auxiliary[0,0,:,hv] if auxiliary is not None else None\n',1)
    needle='                    if bool((candidate < best * 0.99).all()):'
    assert source.count(needle)==1
    source=source.replace(needle,"""                    original=bool((candidate < best * 0.99).all())
                    rescue=(not original and auxiliary is not None and
                            bool((candidate<=best*1.005).all()) and
                            bool((auxiliary[bi,vi,:,hv]<best_aux*.98).all()))
                    if original or rescue:
                        if rescue:rescues+=1
                        if auxiliary is not None:best_aux=auxiliary[bi,vi,:,hv]
""".rstrip())
    source=source.replace('        JOINT_DEBUG.update(balance_heads=int(bmask.sum()), kernel_heads=int(vmask.sum()),','        JOINT_DEBUG.update(rescues=rescues, balance_heads=int(bmask.sum()), kernel_heads=int(vmask.sum()),',1)
    return source+'\n'+HELPER


def frontier_source(source):
    begin=source.index('        for hv in range(kh):\n            best = costs[0, 0, :, hv]')
    end=source.index('        answer = dict(states)',begin)
    replacement='''        for hv in range(kh):
            candidates=costs[:,:,:,hv].reshape(-1,2)
            baseline=candidates[0]
            eligible=(candidates<baseline[None]*.99).all(1)
            eligible[0]=True
            score=candidates.mean(1).masked_fill(~eligible,float('inf'))
            winner=int(score.argmin())
            bmask[hv],vmask[hv]=bool(winner//2),bool(winner%2)
'''
    return source[:begin]+replacement+source[end:]
