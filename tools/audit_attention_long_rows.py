"""Long-row boundary audit; diagnostic scores never enter the main estimator.

Use cached checkpoint projections and modeled hidden states, full Qwen Q-head
layout, and DeepSeek TP-local layout. Quantize full tensors; sample query rows
only AFTER quantization for bounded-memory output-error diagnostics.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from build_model_proxy_dataset import MODELS, build_attention_group
from evaluate_local import mse, score
from hif4_quant import dequantize_hif4, dequantize_nvfp4, quantize_to_hif4
from probe_persistent_covariance import load
from self_check import validate_frozen_state, validate_hif4_params


def sampled_output(tensors, qh, kh, dim, indices):
    q=tensors['q'][indices].reshape(-1,qh,dim).transpose(0,1).float()
    k=tensors['k'].reshape(-1,kh,dim).transpose(0,1).float()
    v=tensors['v'].reshape(-1,kh,dim).transpose(0,1).float()
    k=k.repeat_interleave(qh//kh,0)
    v=v.repeat_interleave(qh//kh,0)
    return torch.softmax(q@k.transpose(1,2)/dim**0.5,-1)@v


def validate_chunks(params, shape, tag):
    # All rows checked, not just scoring queries. Bound float64 checker scratch.
    count=0
    for start in range(0,shape[0],256):
        end=min(start+256,shape[0])
        errors=validate_hif4_params({k:v[start:end] for k,v in params.items()},
                                   (end-start,shape[1]),f'{tag}/{start}:{end}')
        assert not errors,errors
        count+=1
    return count


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--versions',default='307,311,312')
    p.add_argument('--rows',default='1,1024,2048,4096,8192')
    p.add_argument('--query-rows',type=int,default=32)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    versions=list(map(int,args.versions.split(',')))
    report={'dataset_type':'checkpoint-projection + modeled hidden states; independent long-row boundary diagnostic',
            'seed':20264903,'rows':list(map(int,args.rows.split(','))),
            'query_rows_for_score':args.query_rows,'all_output_rows_format_checked':True,
            'cpu_threads':torch.get_num_threads(),'models':[],
            'warning':'not added to main bank, not online scores, no aggregate online timing extrapolation'}
    for index,model in enumerate(MODELS):
        group=build_attention_group(model,20264903+index*1000,'online-shape',
              tuple(report['rows']),'checkpoint-projection',ROOT/'local_model_tensor_cache',.05,128,4.)
        qh,kh,dim=[group[k] for k in ['q_num_heads','kv_num_heads','head_dim']]
        record={'model':model.repo_id,'revision':model.revision,'q_heads':qh,'kv_heads':kh,'head_dim':dim,
                'versions':{},'baseline_mse':[]}
        report['models'].append(record)
        modules={v:load(v) for v in versions};states={}
        for version,module in modules.items():
            started=time.perf_counter()
            states[version]=module.hif4_calibration_attention(group['calib'],qh,kh,dim)
            seconds=time.perf_counter()-started
            errors=validate_frozen_state(states[version],str(version));assert not errors,errors
            record['versions'][str(version)]={'calibration_seconds':seconds,'state_valid':True,'tests':[]}
        for sample in group['test']:
            n=sample['q'][0].shape[0]
            ids=torch.arange(0,n,max(1,(n+args.query_rows-1)//args.query_rows))
            original={r:dequantize_nvfp4(*sample[r]).float() for r in 'qkv'}
            baseline={r:dequantize_hif4(quantize_to_hif4(x),tuple(x.shape)) for r,x in original.items()}
            target=sampled_output(original,qh,kh,dim,ids)
            bm=mse(target,sampled_output(baseline,qh,kh,dim,ids))
            record['baseline_mse'].append(bm)
            del baseline
            # Alternate timed version order to reduce systematic warm-up bias.
            order=versions if n in (1,2048,8192) else list(reversed(versions))
            for version in order:
                module=modules[version];player={};times={};checks=0
                for role in 'qkv':
                    started=time.perf_counter()
                    params=getattr(module,'hif4_dynamic_quantize_'+role)(
                        *sample[role],qh if role=='q' else kh,dim,states[version][role+'_state'])
                    times[role]=time.perf_counter()-started
                    checks+=validate_chunks(params,original[role].shape,f'{version}/{n}/{role}')
                    player[role]=dequantize_hif4(params,tuple(original[role].shape))
                    del params
                pm=mse(target,sampled_output(player,qh,kh,dim,ids))
                result={'rows':n,'query_rows_scored':len(ids),'raw_sampled_score':score(bm,pm),
                        'player_mse':pm,'dynamic_seconds':times,'valid_parameter_chunks':checks}
                record['versions'][str(version)]['tests'].append(result)
                print(model.name,version,n,result['raw_sampled_score'],times,flush=True)
                del player
            Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
            del original,target
        del group,states,modules


if __name__=='__main__':main()
