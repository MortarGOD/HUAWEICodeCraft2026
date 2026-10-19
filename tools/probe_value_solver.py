"""Value solver and attention-concentration diagnostics on fixed real projections."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import evaluate_attention
from probe_persistent_covariance import load


def configured(name):
    if name in ("v309", "v310"):
        return load(int(name[1:]))
    m = load(310)
    if name.startswith("polish"):
        parent = m.hif4_dynamic_quantize_v
        steps = int(name.removeprefix("polish"))
        def polish(q,s,nh,d,state):
            p = parent(q,s,nh,d,state)
            x = m.dequantize_nvfp4(q,s).float()
            T,C = x.shape
            if T <= 1: return p
            step = (m._params_unit_flat(p)*0.25).T.contiguous()
            code = torch.round(m._deq_params(p).T / step).contiguous()
            error = x.T.contiguous() - code*step
            summed = error.sum(1)
            curvature = 2*step.square()
            rows = torch.arange(C)
            for _ in range(steps):
                gradient = summed[:,None] + error
                direction = torch.where(gradient>0,1.0,-1.0)
                gain = curvature - 2*step*gradient.abs()
                gain = gain.masked_fill(torch.where(gradient>0,code>=7,code<=-7),float("inf"))
                idx = gain.argmin(1)
                active = gain[rows,idx] < 0
                shift = direction[rows,idx]*active
                delta = shift*step[rows,idx]
                code[rows,idx] += shift
                error[rows,idx] -= delta
                summed -= delta
            return m._values_to_params((code*step).T.contiguous(),p)
        m.hif4_dynamic_quantize_v = polish
        return m
    def dynamic(q, s, nh, d, state):
        x = m.dequantize_nvfp4(q,s).float()
        p = m._dyn_table(x,None,False)
        T = x.shape[0]
        if T <= 1: return p
        unit = m._params_unit_flat(p)
        best = m._deq_params(p)
        lam = 1.0/T
        def consider(value):
            error = x-value
            mean = error.mean(0)
            return mean.square()+lam*error.square().mean(0),mean
        best_cost, shift = consider(best)
        if name.startswith("bisect"):
            bound = unit.amax(0)*0.25
            lower,upper = -bound,bound
            steps = int(name.removeprefix("bisect"))
        else:
            steps = int(name.removeprefix("iter"))
        for i in range(steps):
            if name.startswith("bisect"):
                shift = (lower+upper)*0.5
            value = torch.round((x+shift)/unit*4).clamp(-7,7)*(unit*0.25)
            cost,mean = consider(value)
            accepted = cost<best_cost
            best = torch.where(accepted[None],value,best)
            best_cost = torch.minimum(cost,best_cost)
            if name.startswith("bisect"):
                positive = mean-lam*shift > 0
                lower = torch.where(positive,shift,lower)
                upper = torch.where(positive,upper,shift)
            else:
                shift = shift + 0.75*mean
        return m._values_to_params(best,p)
    m.hif4_dynamic_quantize_v = dynamic
    return m


def sharpen(groups, factor):
    if factor == 1: return groups
    def sample(s):
        result = dict(s)
        for r in ("q","k"):
            result[r] = [s[r][0],s[r][1]*factor]
        return result
    return [dict(g,calib=[sample(s) for s in g["calib"]],
                 test=[sample(s) for s in g["test"]]) for g in groups]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--configs",default="v310,bisect8,bisect12,iter16,iter32")
    p.add_argument("--factors",default="1")
    p.add_argument("--groups-per-model",type=int,default=5)
    p.add_argument("--offset",type=int,default=10)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    all_groups=torch.load(ROOT/"local_fitted_models_covariance_v1/attn.pt",weights_only=False)
    groups,counts=[],{}
    for g in all_groups:
        key=g["source_model"]; i=counts.get(key,0);counts[key]=i+1
        if args.offset<=i<args.offset+args.groups_per_model: groups.append(g)
    report={"dataset":"local_fitted_models_covariance_v1","offset":args.offset,
            "groups":len(groups),"warning":"factors other than 1 are deliberate distribution stress tests, not new online estimates","results":{}}
    for factor in map(float,args.factors.split(",")):
        samples=sharpen(groups,factor)
        for name in args.configs.split(","):
            m=configured(name)
            with contextlib.redirect_stdout(io.StringIO()):
                scores,secs=evaluate_attention(m,samples)
            key=f"{name}@{factor:g}"
            report["results"][key]={"score":sum(scores),"seconds":secs,"cases":scores}
            Path(args.output).write_text(json.dumps(report,indent=2)+"\n")
            print(key,sum(scores),secs,flush=True)


if __name__=="__main__": main()
