"""Measure GPTQ block scheduling without changing its mathematical objective."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import evaluate_attention
from probe_persistent_covariance import load


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--blocks",default="128,64,32,16,8")
    p.add_argument("--groups-per-model",type=int,default=2)
    p.add_argument("--offset",type=int,default=0)
    args=p.parse_args()
    bank=torch.load(ROOT/"local_fitted_models_covariance_v1/attn.pt",weights_only=False)
    groups=bank[args.offset:args.offset+args.groups_per_model]+bank[25+args.offset:25+args.offset+args.groups_per_model]
    results={"parent":311,"groups":len(groups),"offset":args.offset,
             "threads":torch.get_num_threads(),"results":{},
             "warning":"same exact-arithmetic GPTQ, not guaranteed bitwise equal after block GEMM reassociation"}
    for block in map(int,args.blocks.split(',')):
        m=load(311);m.GPTQ_BLOCK=block
        stats={"calls":0,"seconds":0.0};original=m._gptq_quantize_batched
        def traced(*a,**kw):
            started=time.perf_counter()
            result=original(*a,**kw)
            stats["seconds"]+=time.perf_counter()-started;stats["calls"]+=1
            return result
        m._gptq_quantize_batched=traced
        with contextlib.redirect_stdout(io.StringIO()):
            scores,seconds=evaluate_attention(m,groups)
        results['results'][str(block)]={"score":sum(scores),"seconds":seconds,"cases":scores,"gptq":stats}
        Path(args.output).write_text(json.dumps(results,indent=2)+'\n')
        print(block,sum(scores),seconds,stats,flush=True)


if __name__=="__main__":main()
