"""Large Q/K regrouping and cross-channel mixing, with actual deployed guard."""
import argparse
import ast
import contextlib
import copy
import io
import json
import torch
from probe_v319_value import ROOT, load_solution
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result, validate_hif4_params


def install(m):
    source = (ROOT/'tools/learned_rotation_core.py').read_text()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in ('_lr_states', '_lr_guard'):
            exec(compile(ast.get_source_segment(source,node), 'regroup-helper', 'exec'), m.__dict__)
    m.LR_DEBUG = {}
    m._LR_HEAD_CAP = 0


def transforms(m, samples, states, qh, kh, d, family='orthogonal'):
    energies = []
    for role, nh in (('q',qh),('k',kh)):
        values = []
        for s in samples[:-1][-2:]:
            t=s[role][0].shape[0]
            ix=torch.linspace(0,t-1,min(128,t)).long()
            raw=m.dequantize_nvfp4(s[role][0][ix],s[role][1][ix]).float()
            x=m._kt_input(raw,states[role+'_state'],nh,d,role).reshape(-1,kh,nh//kh,d)
            values.append(x.square().mean((0,2)))
        energies.append(torch.stack(values).mean(0).clamp_min(1e-20).sqrt())
    q,k=energies
    eye=torch.eye(d).expand(kh,d,d)
    if family=='nonorthogonal':
        scale=(k/q).sqrt().clamp(.25,4.)
        result={f'balance_{power}':torch.diag_embed(scale.pow(power)) for power in (.5,1.,-1.)}
        for role,power in (('q',.5),('k',-.5)):
            u=states[role+'_state']['u'].float()
            inverse=torch.linalg.solve_triangular(u,torch.eye(d).expand_as(u),upper=True)
            h=inverse@inverse.transpose(1,2)
            h=h.reshape(kh,-1,d,d).mean(1)
            eigen,vectors=torch.linalg.eigh((h+h.transpose(1,2))*.5)
            eigen=eigen.clamp_min(eigen.amax(-1)[:,None]*.001)
            gain=eigen.pow(power)
            gain=(gain/gain.log().mean(-1).exp()[:,None]).clamp(.25,4.)
            result['whiten_'+role]=(vectors*gain[:,None])@vectors.transpose(1,2)
        return result
    orders = {
        'key_sorted':k.argsort(-1),
        'joint_sorted':(q.log()+k.log()).argsort(-1),
        'query_sorted':q.argsort(-1),
    }
    result={name:eye.gather(2,ix[:,None].expand(kh,d,d)).contiguous() for name,ix in orders.items()}
    # Cross-scale pairs: mix largest with smallest instead of infinitesimal STE updates.
    order=orders['joint_sorted']
    mix=torch.zeros_like(eye)
    for head in range(kh):
        lo=order[head,:d//2];hi=order[head,d//2:].flip(0)
        cols=torch.arange(d//2)*2
        mix[head,lo,cols]=2**-.5;mix[head,hi,cols]=2**-.5
        mix[head,lo,cols+1]=2**-.5;mix[head,hi,cols+1]=-2**-.5
    result['cross_pair_mix']=mix
    return result


def nonorthogonal_states(m,states,r,qh,kh,d):
    ri=torch.linalg.inv(r).transpose(1,2)
    result=dict(states)
    for role,transform,other in (('q',r,ri),('k',ri,r)):
        old=states[role+'_state'];st=dict(old)
        if 'joint_transform' in old:base=old['joint_transform'].float()
        else:
            base=m._make_R(d,int(old.get('rseed') or 0)) if old.get('rot') else torch.eye(d)
            base=base[None].expand(kh,d,d)
        st['joint_transform']=(base@transform).contiguous()
        u=old['u'].float();tr=transform.repeat_interleave(u.shape[0]//kh,0)
        inv=tr.transpose(1,2)@(u.transpose(1,2)@u)@tr
        st['u']=torch.linalg.cholesky((inv+inv.transpose(1,2))*.5,upper=True).contiguous()
        if 'H' in old:
            h=old['H'].float();ot=other.repeat_interleave(h.shape[0]//kh,0)
            st['H']=(ot.transpose(1,2)@h@ot).to(old['H'].dtype).contiguous()
        result[role+'_state']=st
    return result, (r@ri.transpose(1,2)-torch.eye(d)).abs().max().item()


def metric_states(m,states,name,qh,kh,d):
    result=dict(states)
    for role in 'qk':
        old=states[role+'_state'];st=dict(old)
        if 'joint_transform' not in st:
            r=m._make_R(d,int(old.get('rseed') or 0)) if old.get('rot') else torch.eye(d)
            st['joint_transform']=r[None].expand(kh,d,d).contiguous()
        if name.startswith('both') or role=='q':
            u=old['u'].float()
            inverse=torch.linalg.solve_triangular(u,torch.eye(d).expand_as(u),upper=True)
            h=inverse@inverse.transpose(1,2)
            if name=='both_shrink':h=.5*h+.5*torch.diag_embed(h.diagonal(dim1=-2,dim2=-1))
            e,v=torch.linalg.eigh((h+h.transpose(1,2))*.5)
            e=e.clamp_min(e.amax(-1)[:,None]*1e-5)
            power=2. if name=='q_sharp' else (.5 if name.endswith('flat') else 1.)
            e=e.pow(power);e=e/e.mean(-1)[:,None]
            h=(v*e[:,None])@v.transpose(1,2)
            hi=(v*e.reciprocal()[:,None])@v.transpose(1,2)
            st['u']=torch.linalg.cholesky((hi+hi.transpose(1,2))*.5,upper=True).contiguous()
            if 'H' in old:st['H']=h.to(old['H'].dtype).contiguous()
        result[role+'_state']=st
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--groups',default='0,1,2,14,19,49')
    parser.add_argument('--bank',default='local_fitted_models_attention_feedback_v2')
    parser.add_argument('--output',default='docs/v326-regroup-screen.json')
    parser.add_argument('--family',choices=('orthogonal','nonorthogonal','metric'),default='orthogonal')
    parser.add_argument('--modes',default='',help='Optional named arms; never choose per test case.')
    a=parser.parse_args();torch.set_num_threads(4)
    bank=torch.load(ROOT/a.bank/'attn.pt',weights_only=False)
    m=load_solution(ROOT/'solution/solution_v324.py');install(m)
    parent_cal=m.hif4_calibration_attention
    checks={'states':0,'returns':0}
    for role in 'qkv':
        original=getattr(m,'hif4_dynamic_quantize_'+role)
        def checked(*args,_original=original):
            answer=_original(*args)
            assert not validate_hif4_params(answer,args[0].shape,'regroup')
            checks['returns']+=1
            return answer
        setattr(m,'hif4_dynamic_quantize_'+role,checked)
    rows=[]
    for gi in map(int,a.groups.split(',')):
        g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        states=parent_cal(g['calib'],qh,kh,d)
        m.hif4_calibration_attention=lambda *_,_st=states:copy.deepcopy(_st)
        with contextlib.redirect_stdout(io.StringIO()):baseline,_=evaluate_attention(m,[g])
        row=dict(group=gi,baseline=baseline,modes={})
        if not all('u' in states[r+'_state'] for r in 'qk'):
            row['skipped']='missing GPTQ metric';rows.append(row);continue
        candidates=({name:None for name in ('q_flat','both_flat','q_sharp','both_shrink')}
                    if a.family=='metric' else transforms(m,g['calib'],states,qh,kh,d,a.family))
        for name,rotation in candidates.items():
            if a.modes and name not in a.modes.split(','):continue
            if a.family=='metric':
                proposed=metric_states(m,states,name,qh,kh,d);residual=0.
            elif a.family=='orthogonal':
                residual=(rotation.transpose(1,2)@rotation-torch.eye(d)).abs().max().item()
                proposed=m._lr_states(states,rotation,qh,kh,d)
            else:proposed,residual=nonorthogonal_states(m,states,rotation,qh,kh,d)
            assert residual<1e-4
            m.LR_DEBUG.clear()
            chosen=m._lr_guard(g['calib'][-1],states,proposed,qh,kh,d)
            assert not validate_attention_calibration_result(chosen,'regroup')
            checks['states']+=1
            m.hif4_calibration_attention=lambda *_,_st=chosen:copy.deepcopy(_st)
            with contextlib.redirect_stdout(io.StringIO()):score,_=evaluate_attention(m,[g])
            row['modes'][name]=dict(delta=sum(score)-sum(baseline),scores=score,
                continuous_invariance_error=residual,debug=copy.deepcopy(m.LR_DEBUG))
        rows.append(row)
        print(gi,{k:(round(v['delta'],6),v['debug'].get('adopted')) for k,v in row['modes'].items()},flush=True)
    names=sorted({k for r in rows for k in r['modes']})
    report=dict(complete=True,bank=a.bank,family=a.family,rows=rows,checks=checks,
        totals={k:sum(r['modes'].get(k,{}).get('delta',0.) for r in rows) for k in names},
        warning='Checkpoint projections with modeled activations; no online score or timing prediction. Four separate arms, never test-selected mixing.')
    (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(report['totals'],checks,flush=True)


if __name__=='__main__':main()
