"""HiF4 four-value floor/ceil enumeration instead of a few RTN rays."""
from pathlib import Path
from probe_v319_value import ROOT

def source(last_passes=0):
    s=(ROOT/'solution/solution_v322.py').read_text()
    s=s.replace('    alternate_values: torch.Tensor | None = None,','    alternate_values: torch.Tensor | None = None,\n    qubo_refinement: bool = False,')
    a=s.index('            proposal = old[:, None] + lookup_steps * direction[:, None]')
    b=s.index('            candidate_value = torch.cat((old[:, None], proposal_value), dim=1)',a)
    original=s[a:b]
    cond='qubo_refinement' if not last_passes else f'qubo_refinement and _ >= micro_passes - {last_passes}'
    new=f'''            use_qubo = {cond}
            if use_qubo:
                current_lv3 = result_lv3[:, block].to(torch.float32)
                alternate_lv3 = 3.0-current_lv3
                scale_bank = torch.stack((hierarchy_scale[:,block],hierarchy_base[:,block]*alternate_lv3),dim=1)
                center = old + direction
                lower = torch.floor(center[:,None]/scale_bank*4.0)
                patterns = torch.arange(16,device=old.device)[:,None]
                bits = ((patterns >> torch.arange(4,device=old.device)[None]) & 1).to(torch.float32)
                signed_code = (lower[:,:,None]+bits[None,None,:,None]).clamp(-7.,7.).reshape(flat_rows,32,16,4)
                mantissa_code = signed_code.abs()
                proposal_sign = torch.sign(signed_code)
                expanded_scale = scale_bank[:,:,None].expand(-1,-1,16,-1,-1).reshape(flat_rows,32,16,1)
                proposal_value = signed_code * expanded_scale / 4.0
            else:
'''+''.join('    '+line+'\n' for line in original.splitlines())
    s=s[:a]+new+s[b:]
    a=s.index('            proposal_lv3 = torch.stack(',a)
    b=s.index('            lv3_bank = torch.cat(',a)
    old=s[a:b]
    new=old.replace('-1, -1, 3, -1, -1','-1, -1, (16 if use_qubo else 3), -1, -1').replace('flat_rows, 6, 16, 1','flat_rows, (32 if use_qubo else 6), 16, 1')
    s=s[:a]+new+s[b:]
    point='''            extra_refinement=(shape == -1),
            alternate_values=('''
    s=s.replace(point,'''            extra_refinement=(shape == -1),
            qubo_refinement=bool(activation_state.get("linear_qubo", False)),
            alternate_values=(''')
    return s

if __name__=='__main__':
    import argparse,copy,json,time
    import torch
    from probe_v319_value import load_solution
    from hif4_quant import dequantize_hif4,quantize_to_hif4
    from self_check import validate_hif4_params
    p=argparse.ArgumentParser();p.add_argument('--groups',default='0,4,6,7,9');p.add_argument('--last',type=int,default=0);p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2');p.add_argument('--output',default='docs/v324-qubo-screen.json');args=p.parse_args()
    torch.set_num_threads(4);m=load_solution(ROOT/'solution/solution_v322.py');new=load_solution(ROOT/'solution/solution_v322.py');exec(compile(source(args.last),'qubo','exec'),new.__dict__)
    bank=torch.load(ROOT/args.dataset/'linear.pt',weights_only=False);rows=[]
    for gi in map(int,args.groups.split(',')):
        g=bank[gi];w=m.dequantize_nvfp4(*g['weight']).float();bw=dequantize_hif4(quantize_to_hif4(w),w.shape).float();parent=m.hif4_calibration_and_quantize_weight(*g['weight'],g['calib_activation_list']);b=m._linear_score_decode(parent['weight_params'])
        state=copy.deepcopy(parent['activation_state']);state['linear_qubo']=True
        for ti,packed in enumerate(g['test_activation_list']):
            x=m.dequantize_nvfp4(*packed).float();y=x@w.T;xb=dequantize_hif4(quantize_to_hif4(x),x.shape).float();bm=float((y-xb@bw.T).square().mean());row=dict(group=gi,test=ti,scores={},seconds={})
            for name,mod,st in (('parent',m,parent['activation_state']),('qubo',new,state)):
                start=time.perf_counter();pa=mod.hif4_dynamic_quantize_activation(*packed,st);row['seconds'][name]=time.perf_counter()-start;assert not validate_hif4_params(pa,x.shape,name);z=m._linear_score_decode(pa);row['scores'][name]=1-float((y-z@b.T).square().mean())/bm
            row['delta']=row['scores']['qubo']-row['scores']['parent'];rows.append(row)
        print(gi,sum(r['delta'] for r in rows),flush=True)
    (ROOT/args.output).write_text(json.dumps(dict(complete=True,last=args.last,rows=rows,delta=sum(r['delta'] for r in rows),score_prediction_allowed=False),indent=2)+'\n')
