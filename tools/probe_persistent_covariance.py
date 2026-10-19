"""Mechanistic covariance scenarios from cached checkpoint projections.

Not real captured activations; seeds and all attempted scenarios are reported.
"""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import evaluate_attention
from hif4_quant import dequantize_nvfp4
from build_model_proxy_dataset import MODELS, build_attention_group


def load(version):
    spec = importlib.util.spec_from_file_location(
        f"probe{version}", ROOT / f"solution/solution_v{version}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stats(group):
    heads, dim = group["kv_num_heads"], group["head_dim"]
    def gram(samples):
        pieces = [dequantize_nvfp4(*s["k"]).float().reshape(-1, heads, dim)
                  for s in samples]
        x = torch.cat(pieces).permute(1, 0, 2)
        return torch.bmm(x.transpose(1, 2), x) / x.shape[1]
    c, t = gram(group["calib"][:-1]), gram(group["test"])
    norm2 = c.square().sum((-2, -1))
    diag = c.diagonal(dim1=-2, dim2=-1)
    return {
        "k_effective_rank": float((diag.sum(-1).square() / norm2).mean()),
        "k_offdiag_fraction": float((1 - diag.square().sum(-1) / norm2).mean()),
        "k_cal_test_cosine": float(((c*t).sum((-2,-1)) /
            (norm2*t.square().sum((-2,-1))).sqrt()).mean()),
    }


def run(groups, versions):
    result = {}
    for version in versions:
        m = load(version)
        flags = {"gq": 0, "rf_q": 0, "rf_k": 0, "rot": 0, "qs": 0}
        original = m.hif4_calibration_attention
        def capture(*args):
            st = original(*args)
            q, k = st["q_state"], st["k_state"]
            if isinstance(q, dict):
                flags["gq"] += bool(q.get("gq", 0))
                flags["rf_q"] += bool(q.get("rf", 0))
                flags["rot"] += bool(q.get("rot", 0))
                flags["qs"] += "qs" in q
            if isinstance(k, dict):
                flags["rf_k"] += bool(k.get("rf", 0))
            return st
        m.hif4_calibration_attention = capture
        with contextlib.redirect_stdout(io.StringIO()):
            scores, seconds = evaluate_attention(m, groups)
        result[str(version)] = {"score": sum(scores), "seconds": seconds,
                                "flags": flags, "cases": scores}
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output", default="docs/v308-covariance-sweep.json")
    p.add_argument("--seed", type=int, default=20260903)
    p.add_argument("--versions", default="285,303,305,306,307")
    p.add_argument("--groups", type=int, default=1)
    p.add_argument("--rows", default="1,10,64,128,256")
    p.add_argument("--configs", default="0:0:0.8,16:2:0.05,32:4:0.05,64:4:0.05,128:4:0.05")
    args=p.parse_args()
    reports=[]
    for config in args.configs.split(","):
        rank, strength, drift=map(float,config.split(":"))
        groups=[build_attention_group(model,args.seed+mi*1000+gi*37,
                    "correlation",tuple(map(int,args.rows.split(","))),
                    "checkpoint-projection",ROOT/"local_model_tensor_cache",
                    drift,int(rank),strength)
                for mi,model in enumerate(MODELS) for gi in range(args.groups)]
        result=run(groups,list(map(int,args.versions.split(","))))
        record={"config":config,"stats":[stats(g) for g in groups],"results":result}
        reports.append(record)
        print(config, {v:round(r["score"],5) for v,r in result.items()},
              {v:r["flags"] for v,r in result.items()},flush=True)
    Path(args.output).write_text(json.dumps({"seed":args.seed,"rows":args.rows,
        "groups_per_model":args.groups,"provenance":"checkpoint projections + modeled hidden states",
        "reports":reports},indent=2)+"\n")


if __name__ == "__main__":
    main()
