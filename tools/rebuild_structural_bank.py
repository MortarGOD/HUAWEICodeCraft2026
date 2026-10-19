"""Deterministic checkpoint projection proxy, not captured forward activations.

No contestant output, benchmark result, or optimized mixture weights are read.
All stochastic structure is applied BEFORE projections and role normalization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from build_model_proxy_dataset import MODELS,_attention_projections,_rms_normalize,_apply_partial_rope
from build_fitted_dataset import quantize_nvfp4
from self_check import _normalize_attention_dataset,_normalize_linear_dataset

PROFILES=[
    dict(name='broad',rank=64,strength=.3,shared=.0,cluster=.0,drift=.0),
    dict(name='persistent',rank=32,strength=2.,shared=.3,cluster=.0,drift=.05),
    dict(name='clustered',rank=16,strength=1.,shared=.5,cluster=1.5,drift=.1),
    dict(name='common_context',rank=16,strength=.7,shared=2.,cluster=.5,drift=.05),
    dict(name='topic_shift',rank=64,strength=2.,shared=.8,cluster=.8,drift=.5),
]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def query_indices(full_q,kv,local_q,gen):
    assert full_q%kv==0 and local_q%kv==0
    width=full_q//kv
    return torch.cat([torch.randperm(width,generator=gen)[:local_q//kv].sort().values+h*width for h in range(kv)])


def project(hidden,model,p,indices):
    t=hidden.shape[0];d=model.head_dim
    if model.name.startswith('qwen'):
        q=(hidden@p['q'].T).view(t,model.q_heads,2*d)[...,:d][:,indices]
        k=(hidden@p['k'].T).view(t,model.kv_heads,d)
        v=(hidden@p['v'].T).view(t,model.kv_heads,d)
        q=_rms_normalize(q)*p['q_norm'].view(1,1,-1)
        k=_rms_normalize(k)*p['k_norm'].view(1,1,-1)
    else:
        latent=_rms_normalize(hidden@p['q_a'].T)*p['q_norm'].view(1,-1)
        q=_rms_normalize((latent@p['q_b'].T).view(t,model.q_heads,d)[:,indices])
        k=(_rms_normalize(hidden@p['kv'].T)*p['kv_norm'].view(1,-1)).view(t,model.kv_heads,d)
    q=_apply_partial_rope(q,model.rotary_dim)
    k=_apply_partial_rope(k,model.rotary_dim)
    if not model.name.startswith('qwen'):v=k
    return {r:quantize_nvfp4(x.reshape(t,-1)) for r,x in zip('qkv',(q,k,v))}


def build_group(index,seed):
    model=MODELS[index%2];profile=PROFILES[(index//2)%5]
    gen=torch.Generator().manual_seed(seed+index*104729)
    p=_attention_projections(model,ROOT/'local_model_tensor_cache')
    width=model.hidden_size;rank=profile['rank']
    source=p['q'] if model.name.startswith('qwen') else p['q_a']
    basis=_rms_normalize(source[torch.randperm(source.shape[0],generator=gen)[:rank]].float())
    common=torch.randn(1,width,generator=gen)
    centers=torch.randn(8,width,generator=gen)
    qh=16 if model.name.startswith('qwen') and index%10==0 else 8
    indices=query_indices(model.q_heads,model.kv_heads,qh,gen)
    lengths=(64,128,256,512,1024,2048)
    def phase(test):
        order=torch.randperm(5,generator=gen).tolist()
        counts=[lengths[i+int((index+int(test))%3==0)] for i in order]
        out=[]
        for n in counts:
            h=torch.randn(n,width,generator=gen)
            h+=(torch.randn(n,rank,generator=gen)@basis)*(profile['strength']/rank**.5)
            h+=profile['shared']*common
            h+=profile['cluster']*centers[torch.randint(8,(n,),generator=gen)]
            # Prompt-level topic variation, not independent Q/K/V distortion.
            h+=profile['drift']*torch.randn(1,width,generator=gen)
            out.append(project(_rms_normalize(h),model,p,indices))
        return out,counts
    calib,cl=phase(False);test,tl=phase(True)
    return dict(key=f'structural-v3/{index}',source_model=model.repo_id,source_revision=model.revision,
        q_num_heads=qh,kv_num_heads=model.kv_heads,head_dim=model.head_dim,attn_type='gqa',
        population_profile=profile,source_query_indices=indices.tolist(),calibration_lengths=cl,
        test_lengths=tl,calib=calib,test=test)


def main():
    a=argparse.ArgumentParser();a.add_argument('--output',default='local_fitted_models_structural_v3_candidate')
    a.add_argument('--seed',type=int,default=2026090601);a.add_argument('--groups',type=int,default=50)
    args=a.parse_args();torch.set_num_threads(4)
    dest=ROOT/args.output
    if dest.exists():raise ValueError('output exists; never overwrite an unvalidated bank')
    dest.mkdir()
    bank=[]
    for i in range(args.groups):
        g=build_group(i,args.seed);bank.append(g)
        print(i,g['population_profile']['name'],g['test_lengths'],flush=True)
    _normalize_attention_dataset(bank)
    for g in bank:
        assert len(g['calib'])==len(g['test'])==5
        if g['source_model'].startswith('deepseek'):
            for s in g['calib']+g['test']:
                assert all(torch.equal(a,b) for a,b in zip(s['k'],s['v']))
        else:
            width=MODELS[0].q_heads//g['kv_num_heads']
            assert [i//width for i in g['source_query_indices']]==[j//(g['q_num_heads']//g['kv_num_heads']) for j in range(g['q_num_heads'])]
    torch.save(bank,dest/'attn.pt')
    linear=ROOT/'local_fitted_models_attention_feedback_v2/linear.pt'
    lg=torch.load(linear,weights_only=False);_normalize_linear_dataset(lg)
    shutil.copy2(linear,dest/'linear.pt')
    manifest=dict(schema='structural-v3',seed=args.seed,attention_groups=len(bank),attention_cases=len(bank)*5,
        linear_groups=len(lg),linear_cases=sum(len(g['test_activation_list']) for g in lg),
        linear_shapes=[list(g['weight'][0].shape) for g in lg],linear_sha256=digest(linear),
        attn_sha256=digest(dest/'attn.pt'),generator_sha256=digest(Path(__file__)),
        status='structural checks passed; behavioral validation pending',score_prediction_allowed=False,
        profiles=PROFILES,mixture='equal design coverage, NOT inferred online proportions',
        provenance='cached checkpoint projections; modeled shared hidden states, NOT real forward captures',
        fixes=['correct Qwen GQA head selection','preserved DeepSeek K/V identity','no post-projection forced sinks or independent role perturbations','independent calibration/test length order'],
        limitations=['unknown online model/layer mixture','single source layer per model','modeled hidden states and partial RoPE inherited from generator','Linear unchanged and undersized for online; not a performance proxy','length distribution is a coverage prior, not a measured online histogram'])
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
