"""Version-independent input stress populations, scored with the unchanged judge proxy.

All Q/K/V start from cached real checkpoint projections. Transformations model
unobserved token-amplitude/tail structure; they are hypotheses, not captured
model activations. No quantizer output is used to construct a population.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from build_fitted_dataset import quantize_nvfp4
from hif4_quant import dequantize_nvfp4,dequantize_hif4,quantize_to_hif4
from evaluate_local import gqa_attention,mse,score
from probe_persistent_covariance import load


PROFILES = {
    'base': {},
    'gain2': {'gain':2}, 'gain4': {'gain':4}, 'gain8': {'gain':8},
    'amp075': {'sigma':0.75}, 'amp15': {'sigma':1.5},
    'amp2': {'sigma':2},
    'vamp15': {'sigma':1.5,'roles':'v'},
    'kvamp15': {'sigma':1.5,'roles':'kv'},
    'amp15_gain2': {'sigma':1.5,'gain':2},
    'amp15_gain4': {'sigma':1.5,'gain':4},
    'kvamp15_gain4': {'sigma':1.5,'roles':'kv','gain':4},
    'tail8': {'tail':8}, 'tail32': {'tail':32}, 'tail128': {'tail':128},
    'kvtail32': {'tail':32,'roles':'kv'},
    'vtail32': {'tail':32,'roles':'v'},
    'tail32_gain4': {'tail':32,'gain':4},
    'independent_tail32': {'tail':32,'uncoupled_v':True},
    'sink_tail8': {'tail':8,'sink':4},
    'sink_tail32': {'tail':32,'sink':8},
    'sink_tail128': {'tail':128,'sink':16},
    **{f'v_tail{tail}_sink{sink}': {'tail':tail,'roles':'v','sink':sink}
       for tail in (1,2,4,8) for sink in (4,8,16,32)},
    **{f'vmean{strength}': {'vmean':strength} for strength in (1,4,16)},
}


def transform_sample(sample, profile, seed, qh, kh, d):
    if not profile:
        return sample
    gen=torch.Generator().manual_seed(seed)
    raw={r:dequantize_nvfp4(*sample[r]).float() for r in 'qkv'}
    t=raw['v'].shape[0]
    amp=torch.ones(t)
    if profile.get('sigma'):
        amp=(torch.randn(t,generator=gen)*profile['sigma']).exp().clamp(1/64,64)
        amp=amp/amp.square().mean().sqrt()
    if profile.get('tail') and t>1:
        # The same ex-ante rare positions are shared across Q/K/V. They are
        # selected without seeing baseline or candidate quantization errors.
        ix=torch.randperm(t,generator=gen)[:max(1,int(t*0.01))]
        amp[ix]*=profile['tail']
    for r in profile.get('roles','qkv'):
        scale=amp
        if r=='v' and profile.get('uncoupled_v'):
            scale=amp[torch.randperm(t,generator=gen)]
        raw[r]=raw[r]*scale[:,None]
    gain=profile.get('gain',1.0)
    raw['q']*=gain**0.5
    raw['k']*=gain**0.5
    if profile.get('vmean'):
        # A nonzero shared semantic component, selected before quantization.
        raw['v']+=profile['vmean']*raw['v'][:1].clone()
    if profile.get('sink') and t>1:
        # Realize a preferred key through actual Q/K coordinates, not a
        # changed scorer. A fixed common direction induces a key-logit bias.
        strength=float(profile['sink'])
        q=raw['q'].reshape(t,qh,d); k=raw['k'].reshape(t,kh,d)
        q[:,:,0]=d**0.5
        target=amp.argmax()
        k[:,:,0]=0
        k[target,:,0]=strength
    return {r:quantize_nvfp4(raw[r]) for r in 'qkv'}


def transform_group(group, profile, seed):
    qh,kh,d=[group[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
    out=dict(group)
    for phase,offset in (('calib',0),('test',10000)):
        out[phase]=[transform_sample(s,profile,seed+offset+idx*101,qh,kh,d)
                    for idx,s in enumerate(group[phase])]
    out['population_profile']=profile
    out['population_seed']=seed
    return out


def feature_stats(original, baseline, qh,kh,d):
    t=original['v'].shape[0]
    q=original['q'].reshape(t,qh,d).transpose(0,1)
    k=original['k'].reshape(t,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
    p=torch.softmax(q@k.transpose(1,2)/math.sqrt(d),-1)
    leverage=p.square().sum(1).reshape(kh,qh//kh,t).sum(1)
    ev=(original['v']-baseline['v']).reshape(t,kh,d).square().mean(-1).T
    def cosine(a,b):
        aa=a-a.mean(-1,keepdim=True);bb=b-b.mean(-1,keepdim=True)
        return float(((aa*bb).sum(-1)/(aa.norm(dim=-1)*bb.norm(dim=-1)).clamp_min(1e-30)).mean())
    return {'tokens':t,'mean_max_probability':float(p.amax(-1).mean()),
            'effective_attention_tokens':float(1/p.square().sum(-1).mean()),
            'leverage_quant_error_correlation':cosine(leverage,ev),
            'value_token_rms_ratio':float(original['v'].square().mean(-1).sqrt().max()/original['v'].square().mean().sqrt().clamp_min(1e-30))}


def evaluate_value_edges(groups):
    m=load(309);v310=load(310);v311=load(311)
    result={'scores':{str(v):[] for v in (309,310,311)},'features':[],
            'value_objectives':{str(v):[] for v in (309,310,311)}, 'gptq_groups':0}
    start=time.perf_counter()
    for group in groups:
        qh,kh,d=[group[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        states=m.hif4_calibration_attention(group['calib'],qh,kh,d)
        result['gptq_groups']+=int(states['q_state'].get('gq',0))
        for sample in group['test']:
            originals={r:dequantize_nvfp4(*sample[r]).float() for r in 'qkv'}
            baselines={r:dequantize_hif4(quantize_to_hif4(x),x.shape) for r,x in originals.items()}
            q=m._deq_params(m.hif4_dynamic_quantize_q(*sample['q'],qh,d,states['q_state']))
            k=m._deq_params(m.hif4_dynamic_quantize_k(*sample['k'],kh,d,states['k_state']))
            target=gqa_attention(*[originals[r] for r in 'qkv'],qh,kh,d)
            bm=mse(target,gqa_attention(*[baselines[r] for r in 'qkv'],qh,kh,d))
            for version,module in ((309,m),(310,v310),(311,v311)):
                v=module._deq_params(module.hif4_dynamic_quantize_v(*sample['v'],kh,d,None))
                pm=mse(target,gqa_attention(q,k,v,qh,kh,d))
                result['scores'][str(version)].append(score(bm,pm))
                error=originals['v']-v
                value_only=mse(target,gqa_attention(originals['q'],originals['k'],v,qh,kh,d))
                result['value_objectives'][str(version)].append({
                    'uniform_surrogate':float((error.sum(0).square()+error.square().sum(0)).mean()),
                    'true_probability_mse':value_only,'baseline_full_mse':bm,
                    'full_mse':pm})
            result['features'].append(feature_stats(originals,baselines,qh,kh,d))
    result['seconds']=time.perf_counter()-start
    result['mean_edges']={f'{a}->{b}':sum(y-x for x,y in zip(result['scores'][str(a)],result['scores'][str(b)]))/len(result['scores'][str(a)]) for a,b in ((309,310),(310,311))}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--profiles',default=','.join(PROFILES))
    p.add_argument('--groups',default='0,25');p.add_argument('--seed',type=int,default=202609041)
    p.add_argument('--output',default='docs/attention-population-screen.json')
    a=p.parse_args();bank=torch.load(ROOT/'local_fitted_models_covariance_v1/attn.pt',weights_only=False)
    groups=[bank[int(i)] for i in a.groups.split(',')]
    report={'provenance':'real checkpoint projections plus explicitly modeled input transformations',
            'scorer':'unchanged unmasked scaled-dot-product attention and official HiF4 baseline',
            'seed':a.seed,'source_group_indices':a.groups,'profiles':{}}
    for name in a.profiles.split(','):
        config=PROFILES[name]
        transformed=[transform_group(g,config,a.seed+gi*100000) for gi,g in enumerate(groups)]
        result=evaluate_value_edges(transformed)
        report['profiles'][name]={'config':config,**result}
        (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')
        print(name,result['mean_edges'],'gptq',result['gptq_groups'],flush=True)


if __name__=='__main__':main()
