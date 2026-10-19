"""Read-only Linear error accounting. Continuous results are NOT submissions."""
from __future__ import annotations

import time
import torch
from hif4_quant import dequantize_hif4
from probe_linear_scalar_cd import refine


def mean_square(x):
    return float(x.double().square().mean())


def factor_decoder(weight, rtol=None):
    """Column space of B[output,input], with explicit numerical rank policy."""
    b=weight.detach().double().contiguous()
    if not bool(torch.isfinite(b).all()):
        raise ValueError('Nonfinite deployed decoder')
    threshold_ratio=max(b.shape)*torch.finfo(torch.float64).eps if rtol is None else rtol
    if threshold_ratio < 0:
        raise ValueError('rtol must be nonnegative')
    # The production residual decoder has disjoint row supports. Its row
    # orthogonality is exact, so avoid a large redundant SVD in this case.
    if bool(((b!=0).sum(0)<=1).all()):
        energy=b.square().sum(1)
        singular=energy.sqrt()
        tolerance=float(singular.max())*threshold_ratio
        active=singular>tolerance
        values=singular[active]
        core=dict(kind='disjoint_rows',b=b,energy=energy,active=active)
    else:
        u,s,vh=torch.linalg.svd(b,full_matrices=False)
        tolerance=float(s.max())*threshold_ratio
        active=s>tolerance
        values=s[active]
        core=dict(kind='svd',b=b,u=u[:,active],s=values,vh=vh[active])
    rank=int(active.sum())
    core['metadata']=dict(method=core['kind'],shape=list(b.shape),rank=rank,
                          rank_rtol=threshold_ratio,rank_atol=tolerance,
                          full_output_rank=rank==b.shape[0],
                          numerically_truncated=rank<min(b.shape),
                          smallest_retained=float(values.min()) if rank else None,
                          condition_retained=float(values.max()/values.min()) if rank else None)
    return core


def continuous_solution(target, geometry):
    y=target.detach().double()
    if geometry['kind']=='disjoint_rows':
        active=geometry['active']
        scale=torch.where(active,geometry['energy'].clamp_min(torch.finfo(torch.float64).tiny).reciprocal(),torch.zeros_like(geometry['energy']))
        z=(y*scale)@geometry['b']
        projection=y*active
    else:
        coordinates=y@geometry['u']
        z=(coordinates/geometry['s'])@geometry['vh']
        projection=coordinates@geometry['u'].T
    return z,projection


def decompose(target, activation, teacher, geometry, actual_output=None):
    """Two separate identities; teacher error must not be added to geometry."""
    y=target.detach().double()
    z=activation.detach().double()
    fit=z@geometry['b'].T
    continuous,projected=continuous_solution(y,geometry)
    orthogonal=y-projected
    in_space=projected-fit
    total=mean_square(y-fit)
    geometric=mean_square(orthogonal)
    discrete=mean_square(in_space)
    cross=float(2*(orthogonal*in_space).mean())
    delta=teacher.detach().double()-y
    residual=teacher.detach().double()-fit
    teacher_mse=mean_square(delta)
    fit_teacher=mean_square(residual)
    teacher_cross=float(-2*(delta*residual).mean())
    normal=(orthogonal@geometry['b']).norm()
    normal_scale=y.norm()*geometry['b'].norm()
    actual=fit if actual_output is None else actual_output.detach().double()
    actual_mse=mean_square(y-actual)
    report=dict(total_exact_decoder_mse=total,geometric_floor_mse=geometric,
                discrete_and_search_mse=discrete,orthogonality_cross_mse=cross,
                geometric_identity_error=total-geometric-discrete-cross,
                normal_equation_relative=float(normal/normal_scale.clamp_min(1e-300)),
                continuous_reconstruction_mse=mean_square(projected-continuous@geometry['b'].T),
                teacher_bias_mse=teacher_mse,fit_to_teacher_mse=fit_teacher,
                teacher_cross_mse=teacher_cross,
                teacher_identity_error=total-fit_teacher-teacher_mse-teacher_cross,
                actual_output_mse_double_reduction=actual_mse,
                evaluation_output_increment_mse=actual_mse-total,
                evaluation_output_difference_mse=mean_square(actual-fit))
    return report,continuous


def capture_dynamic(solution, packed, state):
    """Capture the actual solver argument without changing solver arithmetic."""
    original=solution._quantize_linear_exact_target
    seen=[]
    def capture(*args,**kwargs):
        seen.append((args[1].detach().clone(),args[2].detach().clone()))
        return original(*args,**kwargs)
    solution._quantize_linear_exact_target=capture
    start=time.perf_counter()
    try:
        params=solution.hif4_dynamic_quantize_activation(*packed,state)
    finally:
        solution._quantize_linear_exact_target=original
    seconds=time.perf_counter()-start
    if state.get('direct_output') is True:
        if seen:raise AssertionError('Unexpected exact-search call in direct decoder')
        raw=solution.dequantize_nvfp4(*packed).float()
        teacher=raw@state['target_weight'].float().T
        internal_decoder=None
        route='direct_output'
    else:
        if len(seen)!=1:
            raise ValueError('Unsupported dynamic path: expected exactly one target capture')
        teacher,internal_decoder=seen[0]
        route='exact_search'
    return params,teacher,internal_decoder,route,seconds


def audit_original_transform(solution,activation,weight,state,teacher):
    """Check captured teacher against the actual carried-transform path."""
    if state.get('direct_output'):
        return dict(route='direct_output',extra_transformed_teacher=False)
    transformed=solution._linear_transform(activation.clone(),state)
    pair=state.get('unrestricted_pair_transform')
    if pair is not None:
        transformed=torch.einsum('rgc,gcd->rgd',transformed.float().reshape(-1,pair.shape[0],pair.shape[-1]),pair.float()).reshape_as(transformed)
    rebuilt=transformed@state['unrestricted_target_weight'].float().T
    return dict(route='exact_search',extra_transformed_teacher=True,
                captured_vs_rebuilt_teacher_mse=mean_square(teacher-rebuilt),
                teacher_weight_dtype=str(state['unrestricted_target_weight'].dtype),
                transform_dtype=str(pair.dtype) if pair is not None else None)


def legal_search_witnesses(solution,params,target,teacher,deployed,continuous,indices,channels=64,output_dtype=torch.float32):
    """Bounded sampled witnesses, not a proof of the discrete global optimum."""
    base={k:v.index_select(0,indices).clone() for k,v in params.items()}
    true=target.index_select(0,indices).float()
    old_teacher=teacher.index_select(0,indices).float()
    shape=(len(indices),deployed.shape[1])
    candidates={'parent':base}
    start=time.perf_counter()
    candidates['true_target_cd']=refine(solution,base,true,deployed,channels,1)
    candidates['teacher_target_cd']=refine(solution,base,old_teacher,deployed,channels,1)
    candidates['continuous_rtn']=solution._quantize_linear_values(continuous.index_select(0,indices).float(),offsets=(-1,0,1,2,3))
    values={name:dequantize_hif4(p,shape).to(output_dtype)@deployed.to(output_dtype).T for name,p in candidates.items()}
    # Report true per-row errors. The lower envelope retains the parent row.
    row_errors={name:(true-output).double().square().mean(-1) for name,output in values.items()}
    errors={name:float(error.mean()) for name,error in row_errors.items()}
    best=torch.stack(list(row_errors.values())).amin(0)
    result=dict(sample_rows=len(indices),indices=indices.tolist(),channel_budget=channels,
                true_mse=errors,best_retaining_parent_mse=float(best.mean()),
                recoverable_witness_mse=errors['parent']-float(best.mean()),
                seconds=time.perf_counter()-start,output_dtype=str(output_dtype),
                scope='Sampled rows only. No inference that remaining gap is search-recoverable.')
    return result,candidates
