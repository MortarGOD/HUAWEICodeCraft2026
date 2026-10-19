"""Research/production source builder; only native HiF4 metadata is returned."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

HELPER='''
def _linear_subset_choice(delta, gradient, gram, local_change, mode):
    # Retain the fused candidate, but repair cross-microgroup interference.
    full_change = 2.0 * (delta * gradient).sum(-1) + torch.einsum("rc,cd,rd->r", delta, gram, delta)
    if mode == 1:
        gain, group = local_change.min(-1)
        only = torch.arange(16, device=delta.device)[None, :] == group[:, None]
        use = gain < full_change
        mask = torch.where(use[:, None], only, torch.ones_like(only))
        return (delta.reshape(-1,16,4)*mask[:,:,None]).reshape_as(delta), mask, torch.minimum(gain,full_change)
    ranks = local_change.argsort(-1).argsort(-1)
    masks = torch.stack([torch.ones_like(ranks,dtype=torch.bool)] + [ranks < k for k in (1,2,4,8)],dim=1)
    changes = (delta.reshape(-1,1,16,4)*masks[:,:,:,None]).flatten(-2,-1)
    costs = 2.0*(changes*gradient[:,None]).sum(-1)+torch.einsum("rnc,cd,rnd->rn",changes,gram,changes)
    best = costs.argmin(-1)
    ix = torch.arange(delta.shape[0],device=delta.device)
    return changes[ix,best],masks[ix,best],costs[ix,best]
'''

def subset_candidate_source(mode=1):
    source=(ROOT/'solution/solution_v322.py').read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_quantize_linear_exact_target')
    body=ast.get_source_segment(source,node)
    body=body.replace('    alternate_values: torch.Tensor | None = None,','    alternate_values: torch.Tensor | None = None,\n    subset_mode: int = 0,')
    old='''            accept = full_change < 0.0
            accepted_delta = torch.where('''
    new='''            subset_mask = None
            if subset_mode:
                local_change = (local_linear + local_quadratic).gather(1, selected_index[:,None]).squeeze(1)
                delta_flat, subset_mask, full_change = _linear_subset_choice(
                    delta_flat, gradient_flat, block_gram[block], local_change, subset_mode)
            accept = full_change < 0.0
            accepted_delta = torch.where('''
    assert body.count(old)==1;body=body.replace(old,new)
    old='''            result_sign[:, block].copy_(torch.where(
                accept[:, None, None], selected_sign, old_sign
            ))'''
    new='''            update_mask = accept[:,None,None]
            if subset_mask is not None:
                update_mask = update_mask & subset_mask[:,:,None]
            result_sign[:, block].copy_(torch.where(
                update_mask, selected_sign, old_sign
            ))'''
    assert body.count(old)==1;body=body.replace(old,new)
    body=body.replace('accept[:, None, None], selected_mant, old_mant','update_mask, selected_mant, old_mant')
    body=body.replace('accept[:, None, None], selected_lv3, old_lv3','update_mask, selected_lv3, old_lv3')
    original=ast.get_source_segment(source,node);source=source.replace(original,body)
    point='''            extra_refinement=(shape == -1),
            alternate_values=('''
    assert source.count(point)==1
    source=source.replace(point,'''            extra_refinement=(shape == -1),
            subset_mode=int(activation_state.get("linear_subset_mode", 0)),
            alternate_values=(''')
    source+=HELPER+f'''
_subset_parent_calibration = hif4_calibration_and_quantize_weight

def hif4_calibration_and_quantize_weight(weight_quant, weight_scale, calib_activation_list):
    result = _subset_parent_calibration(weight_quant, weight_scale, calib_activation_list)
    if result["activation_state"].get("linear_shape") == 1:
        result["activation_state"]["linear_subset_mode"] = {mode}
    return result
'''
    end=source.index('"""',source.index('"""')+3)+3
    source='''"""v324: Linear microgroup subset repair; Attention remains v322.
Fixed calibration decoder; exact current-target quadratic subset choices.
"""'''+source[end:]
    compile(source,'candidate-v324','exec');return source

def candidate_source():
    from probe_v324_gram import source as gram_source
    source=gram_source()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_linear_refine_fixed_decoder')
    old=ast.get_source_segment(source,node)
    body=old.replace('            gram = deployed.T @ deployed','            gram = deployed.T @ deployed\n            decoder_gram_diagonal = gram.diagonal().clone()')
    point='''            return {"weight_params": params, "activation_state": candidate_state}'''
    assert body.count(point)==1
    body=body.replace(point,'''            if teacher.shape[0] >= 2 * teacher.shape[1] and teacher.shape[1] <= 8192:
                # The final candidate was accepted using the unchanged parent path.
                # Restore the exact pre-ridge diagonal; do not subtract a rounded ridge.
                gram.diagonal().copy_(decoder_gram_diagonal)
                if bool(torch.isfinite(gram).all()):
                    candidate_state["linear_full_gram"] = gram.cpu().contiguous()
'''+point)
    source=source.replace(old,body)
    source+='''

_gram_parent_calibration = hif4_calibration_and_quantize_weight

def hif4_calibration_and_quantize_weight(weight_quant, weight_scale, calib_activation_list):
    result = _gram_parent_calibration(weight_quant, weight_scale, calib_activation_list)
    state = result["activation_state"]
    outputs, inputs = weight_quant.shape
    # B^T B replaces repeated large-output residual products, only for up.
    # Bound the additional persistent matrix to 256 MiB (8192 squared FP32).
    if (outputs >= 2 * inputs and inputs <= 8192 and state.get("linear_shape") == 1
            and "linear_full_gram" not in state):
        try:
            deployed = state["unrestricted_deployed_weight"].float()
            gram = deployed.T @ deployed
            if bool(torch.isfinite(gram).all()):
                state["linear_full_gram"] = gram.cpu().contiguous()
        except RuntimeError:
            pass
    return result
'''
    end=source.index('"""',source.index('"""')+3)+3
    source='''"""v324: Input-Gram residual updates for bounded up-projection Linear.
Attention and the fixed Linear decoder retain v322. Objective is unchanged
in exact arithmetic; FP32 update order can alter near-tie decisions.
"""'''+source[end:]
    compile(source,'v324','exec');return source

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
    s=candidate_source()
    if a.write:
        for name in ('solution.py','solution_v324.py'):(ROOT/'solution'/name).write_text(s)
    print('source lines',len(s.splitlines()))
