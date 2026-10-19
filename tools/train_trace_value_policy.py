"""Offline-only shared V policy; true Q/K are training teachers, never features."""
import argparse
import copy
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.verify_attention_campaign import equal
from hif4_quant import dequantize_nvfp4,dequantize_hif4
from self_check import validate_hif4_params


def features(x,y,step):
    rms=x.square().mean(0,keepdim=True).sqrt().clamp_min(1e-8)
    err=y-x
    return torch.stack((x/rms,err/step,step/rms,
        x.mean(0,keepdim=True).expand_as(x)/rms,
        err.mean(0,keepdim=True).expand_as(x)/step,
        x.square().mean(1,keepdim=True).sqrt().expand_as(x)/rms,
        x.sign(),(x.abs()/step).clamp_max(8)/8),-1).clamp(-8,8)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--optimizer',choices=['ste','discrete'],default='ste')
    parser.add_argument('--dataset',default='docs/papers/hif4-public-references/test/datasets/qwen3_5_35b_a3b_v1')
    parser.add_argument('--groups',default='all');parser.add_argument('--evaluate-weights');parser.add_argument('--output')
    args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(440)
    m=load(435,'learn_policy');original=m._dyn_table
    all_groups=torch.load(ROOT/args.dataset/'attn.pt',weights_only=True)
    indices=list(range(len(all_groups))) if args.groups=='all' else list(map(int,args.groups.split(',')))
    bank=[all_groups[i] for i in indices]
    states=[]
    with torch.inference_mode():
        for g in bank:
            states.append(m.hif4_calibration_attention(g['calib'],g['q_num_heads'],g['kv_num_heads'],g['head_dim']))
    def prepare(gi,s):
        g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        with torch.no_grad():
            raw={r:dequantize_nvfp4(*s[r]).float() for r in 'qkv'}
            params={r:getattr(m,'hif4_dynamic_quantize_'+r)(*s[r],qh if r=='q' else kh,d,states[gi][r+'_state']) for r in 'qkv'}
            values={r:dequantize_hif4(params[r],raw[r].shape) for r in 'qkv'}
            pq=m._vr_probs(values['q'],values['k'],qh,kh,d).reshape(kh,-1,len(raw['v']))
            p=m._vr_probs(raw['q'],raw['k'],qh,kh,d).reshape(kh,-1,len(raw['v']))
            x=raw['v'];y=values['v'];step=m._params_unit_flat(params['v'])*.25
            target=p@x.reshape(-1,kh,d).transpose(0,1)
            loss=(pq@y.reshape(-1,kh,d).transpose(0,1)-target).square().mean().clamp_min(1e-15)
        return dict(gi=gi,s=s,kh=kh,d=d,x=x,y=y,step=step,features=features(x,y,step),p=pq,target=target,base=loss,params=params['v'])
    # No test input is accessed before model selection finishes.
    train=[prepare(i,s) for i,g in enumerate(bank) for s in g['calib'][:2]]
    valid=[prepare(i,g['calib'][2]) for i,g in enumerate(bank)]
    net=torch.nn.Sequential(torch.nn.Linear(8,12),torch.nn.Tanh(),torch.nn.Linear(12,1))
    torch.nn.init.zeros_(net[-1].weight);torch.nn.init.zeros_(net[-1].bias)
    def forward(ex,grad=False):
        bias=.5*ex['step']*torch.tanh(net(ex['features']).squeeze(-1))
        delta=bias.detach()
        def hook(x,state,has_scale):
            assert x.shape==delta.shape
            return original(x+delta,state,has_scale)
        try:
            m._dyn_table=hook
            with torch.no_grad():ans=m.hif4_dynamic_quantize_v(*ex['s']['v'],ex['kh'],ex['d'],states[ex['gi']]['v_state'])
        finally:m._dyn_table=original
        assert not validate_hif4_params(ans,ex['x'].shape,'learned V')
        out=dequantize_hif4(ans,ex['x'].shape)
        if grad:out=out+(bias-bias.detach())
        loss=(ex['p']@out.reshape(-1,ex['kh'],ex['d']).transpose(0,1)-ex['target']).square().mean()/ex['base']
        return loss,ans
    for ex in valid:
        _,ans=forward(ex);assert equal(ans,ex['params'])
    optimizer=torch.optim.Adam(net.parameters(),lr=.01)
    history=[];best=1.;best_epoch=0;best_state=copy.deepcopy(net.state_dict())
    if args.evaluate_weights:
        best_state=torch.load(ROOT/args.evaluate_weights,weights_only=True);best_epoch=-1;best=None
    for epoch in range(1,31 if args.optimizer=='ste' and not args.evaluate_weights else 1):
        losses=[]
        for ex in train:
            optimizer.zero_grad();loss,_=forward(ex,True);loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        if epoch%5==0:
            with torch.no_grad():v=sum(float(forward(ex)[0]) for ex in valid)/len(valid)
            history.append({'epoch':epoch,'train_ratio':sum(losses)/len(losses),'validation_ratio':v})
            if v<best:best=v;best_epoch=epoch;best_state=copy.deepcopy(net.state_dict())
            print(history[-1],flush=True)
    if args.optimizer=='discrete' and not args.evaluate_weights:
        with torch.no_grad():
            current=1.
            for epoch,step_size in enumerate((.25,.125,.0625),1):
                accepted=0
                for param in net[-1].parameters():
                    flat=param.view(-1)
                    for index in range(len(flat)):
                        initial=float(flat[index]);chosen=initial
                        for sign in (-1,1):
                            flat[index]=initial+sign*step_size
                            loss=sum(float(forward(ex)[0]) for ex in train)/len(train)
                            if loss<current-1e-7:
                                current=loss;chosen=float(flat[index])
                        flat[index]=chosen
                        accepted+=chosen!=initial
                v=sum(float(forward(ex)[0]) for ex in valid)/len(valid)
                history.append({'epoch':epoch,'step_size':step_size,'accepted':accepted,'train_ratio':current,'validation_ratio':v})
                if v<best:best=v;best_epoch=epoch;best_state=copy.deepcopy(net.state_dict())
                print(history[-1],flush=True)
    net.load_state_dict(best_state)
    rows=[]
    with torch.no_grad():
        for gi,g in enumerate(bank):
            for ti,s in enumerate(g['test']):
                ex=prepare(gi,s);loss,ans=forward(ex)
                rows.append({'group':indices[gi],'test':ti,'output_ratio':float(loss),
                             'changed_values':int((dequantize_hif4(ans,ex['x'].shape)!=ex['y']).sum())})
    report={'completed':True,'optimizer':args.optimizer,'dataset':args.dataset,'groups':indices,'evaluate_weights':args.evaluate_weights,
        'seed':440,'features':8,'hidden':12,'parameters':sum(p.numel() for p in net.parameters()),
        'train_samples':len(train),'validation_samples':len(valid),'best_epoch':best_epoch,'best_validation_ratio':best,
        'history':history,'test':rows,'mean_test_ratio':sum(r['output_ratio'] for r in rows)/len(rows),
        'warning':'Research serial hook only, not submission. Exact legal forward; STE mode uses approximate identity gradients, discrete mode searches 13 output-layer parameters with frozen random features. Parent calibration uses all calibration samples, including policy validation; test untouched until selection. One short-trace model source, not online or cross-source evidence.'}
    suffix='' if args.optimizer=='ste' else '-discrete'
    (ROOT/(args.output or f'docs/trace-value-policy{suffix}.json')).write_text(json.dumps(report,indent=2)+'\n')
    if not args.evaluate_weights:torch.save(best_state,ROOT/f'docs/trace-value-policy{suffix}-weights.pt')
    print({k:v for k,v in report.items() if k not in ('history','test')},flush=True)

if __name__=='__main__':main()
