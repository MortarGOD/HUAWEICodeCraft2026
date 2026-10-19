"""Input-coordinate residual gradient, equivalent in exact arithmetic."""
from probe_v319_value import ROOT

def source():
    s=(ROOT/'solution/solution_v322.py').read_text()
    start=s.index('def _quantize_linear_exact_target(');end=s.index('\ndef _joint_exact_weight_candidate(',start)
    body=s[start:end]
    body=body.replace('    alternate_values: torch.Tensor | None = None,','    alternate_values: torch.Tensor | None = None,\n    full_gram: torch.Tensor | None = None,')
    body=body.replace('    residual = flat_target - current @ deployed_weight.T','    residual = flat_target - current @ deployed_weight.T\n    input_gradient = residual @ deployed_weight if full_gram is not None else None\n    if full_gram is not None:\n        residual = None')
    body=body.replace('residual_gradient = residual @ deployed_block','residual_gradient = residual @ deployed_block if full_gram is None else input_gradient[:,start:stop]')
    body=body.replace('residual_gradient = residual @ deployed_super','residual_gradient = residual @ deployed_super if full_gram is None else input_gradient[:,start:stop]')
    body=body.replace('gradient_flat = residual @ deployed_block','gradient_flat = residual @ deployed_block if full_gram is None else input_gradient[:,start:stop]')
    for delta,block in [('delta_selected','deployed_block'),('accepted_delta','deployed_super'),('accepted_delta','deployed_block')]:
        old=f'residual.add_({delta} @ {block}.T)'
        new=f'input_gradient.add_({delta} @ full_gram[start:stop]) if full_gram is not None else residual.add_({delta} @ {block}.T)'
        body=body.replace(old,new)
    s=s[:start]+body+s[end:]
    point='''            extra_refinement=(shape == -1),
            alternate_values=('''
    s=s.replace(point,'''            extra_refinement=(shape == -1),
            full_gram=activation_state.get("linear_full_gram"),
            alternate_values=(''')
    return s

if __name__=='__main__':
    import argparse,copy,json,time
    import torch
    from probe_v319_value import load_solution
    from hif4_quant import dequantize_hif4,quantize_to_hif4
    from self_check import validate_hif4_params
    p=argparse.ArgumentParser();p.add_argument('--groups',default='0,4,6,7,9');p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2');p.add_argument('--output',default='docs/v324-gram-screen.json');p.add_argument('--full-shape',action='store_true');p.add_argument('--wide-stress',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);m=load_solution(ROOT/'solution/solution_v322.py');new=load_solution(ROOT/'solution/solution_v322.py');exec(compile(source(),'gram','exec'),new.__dict__)
    bank=torch.load(ROOT/args.dataset/'linear.pt',weights_only=False);rows=[]
    if args.full_shape:
        from build_model_proxy_dataset import MODELS,build_linear_group
        layers=[(m,l) for m in MODELS for l in m.linear_layers]
        for gi in map(int,args.groups.split(',')):
            model,layer=layers[gi];bank[gi]=build_linear_group(model,layer,2026090317+gi,'online-shape',(1,10,128,512,1024),'checkpoint',ROOT/'local_model_tensor_cache')
    if args.wide_stress:
        from build_model_proxy_dataset import DEEPSEEK_V4_FLASH,build_linear_group
        from build_fitted_dataset import quantize_nvfp4
        assert args.groups=='0' and not args.full_shape
        model=DEEPSEEK_V4_FLASH;cache=ROOT/'local_model_tensor_cache'
        group=build_linear_group(model,model.linear_layers[0],2026090317,'online-shape',(1,10,128,512,1024),'checkpoint',cache)
        matrix=torch.load(cache/'deepseek-v4-flash-0731--attn_wo_b_tp8.pt',weights_only=True)
        group['weight']=quantize_nvfp4(matrix.T.contiguous().float());bank[0]=group
    for gi in map(int,args.groups.split(',')):
        g=bank[gi];w=m.dequantize_nvfp4(*g['weight']).float();bw=dequantize_hif4(quantize_to_hif4(w),w.shape).float();parent=m.hif4_calibration_and_quantize_weight(*g['weight'],g['calib_activation_list']);b=m._linear_score_decode(parent['weight_params'])
        state=copy.deepcopy(parent['activation_state']);start=time.perf_counter();state['linear_full_gram']=b.float().T@b.float();compile_seconds=time.perf_counter()-start
        for ti,packed in enumerate(g['test_activation_list']):
            x=m.dequantize_nvfp4(*packed).float();y=x@w.T;xb=dequantize_hif4(quantize_to_hif4(x),x.shape).float();bm=float((y-xb@bw.T).square().mean());row=dict(group=gi,test=ti,shape=list(w.shape),rows=len(x),scores={},seconds={},compile_seconds=compile_seconds)
            for name,mod,st in (('parent',m,parent['activation_state']),('gram',new,state)):
                start=time.perf_counter();pa=mod.hif4_dynamic_quantize_activation(*packed,st);row['seconds'][name]=time.perf_counter()-start;assert not validate_hif4_params(pa,x.shape,name);z=m._linear_score_decode(pa);row['scores'][name]=1-float((y-z@b.T).square().mean())/bm
            row['delta']=row['scores']['gram']-row['scores']['parent'];rows.append(row)
        print(gi,sum(r['delta'] for r in rows),{n:sum(r['seconds'][n] for r in rows) for n in ('parent','gram')},flush=True)
    (ROOT/args.output).write_text(json.dumps(dict(complete=True,full_shape=args.full_shape,wide_stress=args.wide_stress,rows=rows,delta=sum(r['delta'] for r in rows),score_prediction_allowed=False),indent=2)+'\n')
