"""Calibration-only attention concentration estimates for Value rounding."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import evaluate_attention
from probe_persistent_covariance import load
from probe_value_solver import sharpen


def module_for(mode):
    m=load(311)
    if mode=="baseline": return m
    parent=m.hif4_calibration_attention
    if mode=="adaptive":
        def calibration(samples,qh,kh,d):
            st=parent(samples,qh,kh,d)
            estimates=[]
            for s in samples[:-1]:
                T=s["q"][0].shape[0]
                if T<=1: continue
                qstride=max(1,(T+31)//32)
                kstride=max(1,(T+127)//128)
                q=m.dequantize_nvfp4(*(part[::qstride] for part in s["q"])).float()
                k=m.dequantize_nvfp4(*(part[::kstride] for part in s["k"])).float()
                q=q.reshape(-1,qh,d).transpose(0,1)
                k=k.reshape(-1,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
                p=torch.softmax(q@k.transpose(1,2)/d**0.5,-1)
                S=p.square().sum(-1).mean(-1)
                coefficient=((k.shape[1]*S-1)/(1-S).clamp_min(1e-6)).clamp(0.25,256)
                estimates.append(coefficient.reshape(kh,qh//kh).mean(1))
            c=torch.stack(estimates).median(0).values if estimates else torch.ones(kh)
            st["v_state"]={"mean_penalty":c.repeat_interleave(d).contiguous()}
            return st
        m.hif4_calibration_attention=calibration

    def dynamic(q,s,kh,d,st):
        x=m.dequantize_nvfp4(q,s).float();T,C=x.shape
        p=m._dyn_table(x,None,False)
        if T<=1:return p
        c=st["mean_penalty"] if mode=="adaptive" else torch.full((C,),float(mode))
        unit=m._params_unit_flat(p)
        best=m._deq_params(p)
        error=x-best
        cost_best=error.mean(0).square()+c/T*error.square().mean(0)
        shift=error.mean(0)
        for _ in range(8):
            proposal=torch.round((x+shift)/unit*4).clamp(-7,7)*(unit*.25)
            error=x-proposal;mu=error.mean(0)
            cost=mu.square()+c/T*error.square().mean(0)
            best=torch.where((cost<cost_best)[None],proposal,best)
            cost_best=torch.minimum(cost,cost_best)
            shift+=mu*.75
        step=(unit*.25).T.contiguous()
        code=torch.round(best.T/step).contiguous()
        error=x.T.contiguous()-code*step
        summed=error.sum(1);curvature=(1+c[:,None])*step.square()
        rows=torch.arange(C)
        for _ in range(8):
            gradient=summed[:,None]+c[:,None]*error
            gain=curvature-2*step*gradient.abs()
            gain.masked_fill_(torch.where(gradient>0,code>=7,code<=-7),float("inf"))
            idx=gain.argmin(1);active=gain[rows,idx]<0
            if not bool(active.any()):break
            move=torch.where(gradient[rows,idx]>0,1.0,-1.0)*active
            delta=move*step[rows,idx]
            code[rows,idx]+=move;error[rows,idx]-=delta;summed-=delta
        return m._values_to_params((code*step).T.contiguous(),p)
    m.hif4_dynamic_quantize_v=dynamic
    return m


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--configs",default="baseline,0.25,4,16,adaptive")
    p.add_argument("--factors",default="1")
    p.add_argument("--groups-per-model",type=int,default=5)
    p.add_argument("--offset",type=int,default=20)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    full=torch.load(ROOT/"local_fitted_models_covariance_v1/attn.pt",weights_only=False)
    groups=[];counts={}
    for g in full:
        key=g["source_model"];i=counts.get(key,0);counts[key]=i+1
        if args.offset<=i<args.offset+args.groups_per_model:groups.append(g)
    report={"dataset":"local_fitted_models_covariance_v1","offset":args.offset,
            "groups":len(groups),"results":{},"warning":"non-unit scale factors are diagnostic perturbations only"}
    for factor in map(float,args.factors.split(",")):
        data=sharpen(groups,factor)
        for mode in args.configs.split(","):
            m=module_for(mode);cs=[];parent=m.hif4_calibration_attention
            def capture(*a):
                s=parent(*a)
                if s["v_state"] is not None:cs.extend(s["v_state"]["mean_penalty"].unique().tolist())
                return s
            m.hif4_calibration_attention=capture
            with contextlib.redirect_stdout(io.StringIO()):scores,seconds=evaluate_attention(m,data)
            key=f"{mode}@{factor:g}"
            report["results"][key]={"score":sum(scores),"seconds":seconds,"cases":scores,"calibration_coefficients":cs}
            Path(args.output).write_text(json.dumps(report,indent=2)+"\n")
            print(key,sum(scores),seconds,cs,flush=True)


if __name__=="__main__":main()
