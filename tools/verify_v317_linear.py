"""Paired cleanup equivalence, fixed-decoder gain and serial interface timing."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import load_solution
from hif4_quant import dequantize_nvfp4, dequantize_hif4, quantize_to_hif4
from self_check import validate_linear_calibration_result, validate_hif4_params
from probe_linear_deployed_joint import tree_equal


def source_audit():
    a = ast.parse((ROOT/'solution/solution_v315.py').read_text())
    b = ast.parse((ROOT/'solution/solution_v317.py').read_text())
    changed = {'_joint_exact_weight_candidate', 'hif4_calibration_and_quantize_weight',
               'hif4_dynamic_quantize_activation'}
    extra = {'_linear_score_decode', '_linear_refine_fixed_decoder'}
    before = [ast.dump(n) for n in a.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name not in changed]
    after = [ast.dump(n) for n in b.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name not in changed|extra]
    assert before == after, 'Unintended function changes, including Attention'
    pa=next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name=='hif4_dynamic_quantize_activation')
    pb=next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name=='hif4_dynamic_quantize_activation')
    assert [ast.dump(n) for n in pa.body[2:]] == [ast.dump(n) for n in pb.body[2:]], 'Up/square dynamic changed'
    forbidden = {'os','pathlib','io','tempfile','shutil','subprocess','socket','ctypes','pickle',
                 'marshal','importlib','zipimport','pkgutil','runpy','mmap','resource','pty','tty','sys','builtins'}
    for n in ast.walk(b):
        if isinstance(n, ast.Import):
            assert all(x.name.split('.')[0] not in forbidden for x in n.names)
        if isinstance(n, ast.ImportFrom):
            assert n.module.split('.')[0] not in forbidden
    return len(before)


@torch.inference_mode()
def main(args):
    audit = source_audit()
    parent = load_solution(ROOT/'solution/solution_v315.py')
    clean = load_solution(ROOT/'solution/solution_v317.py')
    clean._LINEAR_JOINT_ROUNDS = 0
    new = load_solution(ROOT/'solution/solution_v317.py')
    groups = torch.load(ROOT/args.dataset/'linear.pt', weights_only=False, map_location='cpu')
    selected = list(range(len(groups))) if args.groups == 'all' else list(map(int, args.groups.split(',')))
    if args.full_shape:
        from build_model_proxy_dataset import MODELS, build_linear_group
        layers = [(m,l) for m in MODELS for l in m.linear_layers]
        for gi in selected:
            model, layer = layers[gi]
            cache = ROOT/'local_model_tensor_cache'
            assert (cache/f"{model.name}--{layer.name.replace('.', '_')}.pt").exists()
            groups[gi] = build_linear_group(model, layer, 2026090317+gi, 'online-shape',
                                            (1,10,128,512,1024), 'checkpoint', cache)
    if args.wide_stress:
        from build_model_proxy_dataset import DEEPSEEK_V4_FLASH, build_linear_group
        from build_fitted_dataset import quantize_nvfp4
        assert selected == [0] and not args.full_shape
        model=DEEPSEEK_V4_FLASH; cache=ROOT/'local_model_tensor_cache'
        assert (cache/'deepseek-v4-flash-0731--attn_wq_a.pt').exists()
        group=build_linear_group(model, model.linear_layers[0], 2026090317,
                                 'online-shape',(1,10,128,512,1024),'checkpoint',cache)
        matrix=torch.load(cache/'deepseek-v4-flash-0731--attn_wo_b_tp8.pt',weights_only=True)
        group['weight']=quantize_nvfp4(matrix.T.contiguous().float())
        groups[0]=group
    report = dict(complete=False, source_unchanged_functions=audit, full_shape=args.full_shape,
                  dataset=args.dataset, groups=[], validation_checks=0,
                  timing='Serial algorithm calls only; scoring and checks excluded; not a 96-core forecast')
    report['sha256']=hashlib.sha256((ROOT/'solution/solution_v317.py').read_bytes()).hexdigest()
    report['wide_stress'] = args.wide_stress
    if args.wide_stress:
        report['warning']='Transposed real checkpoint W and modeled activations: runtime stress only, NOT an actual model layer or online quality proxy'
    consumed = ['coefficient','permutation','hadamard','linear_shape','unrestricted_pair_transform',
                'unrestricted_target_weight','unrestricted_deployed_weight','unrestricted_continuous_projection',
                'unrestricted_block_gram','unrestricted_block_inverse','unrestricted_micro_inverse']
    for gi in selected:
        group = groups[gi]
        w = dequantize_nvfp4(*group['weight']).float()
        reference_b = dequantize_hif4(quantize_to_hif4(w), tuple(w.shape)).float()
        states = []; timing = {}
        for name, mod in (('parent',parent), ('cleanup',clean), ('joint',new)):
            t = time.perf_counter()
            result = mod.hif4_calibration_and_quantize_weight(*group['weight'], group['calib_activation_list'])
            timing[name] = dict(calibration=time.perf_counter()-t, dynamic=0.)
            assert not validate_linear_calibration_result(result, w.shape, name)
            report['validation_checks'] += 1
            states.append(result)
        assert tree_equal(states[0]['weight_params'], states[1]['weight_params']), ('cleanup weight',gi)
        if states[0]['activation_state'].get('direct_output'):
            assert tree_equal(states[0], states[1])
        else:
            for key in consumed:
                assert tree_equal(states[0]['activation_state'].get(key),states[1]['activation_state'].get(key)), (gi,key)
        cases=[]
        for ti, packed in enumerate(group['test_activation_list']):
            a=dequantize_nvfp4(*packed).float(); target=a@w.T
            base_a=dequantize_hif4(quantize_to_hif4(a), tuple(a.shape)).float()
            bmse=float((target-base_a@reference_b.T).square().mean())
            params=[]; scores=[]
            for (name,mod),cal in zip((('parent',parent),('cleanup',clean),('joint',new)),states):
                t=time.perf_counter()
                qa=mod.hif4_dynamic_quantize_activation(*packed,cal['activation_state'])
                timing[name]['dynamic'] += time.perf_counter()-t
                assert not validate_hif4_params(qa, a.shape, name)
                report['validation_checks'] += 1
                z=dequantize_hif4(qa,tuple(a.shape)); b=dequantize_hif4(cal['weight_params'],tuple(w.shape))
                scores.append(1-float((target-(z@b.T).float()).square().mean())/bmse)
                params.append(qa)
            assert tree_equal(params[0],params[1]), ('cleanup dynamic',gi,ti)
            cases.append(dict(test=ti,parent=scores[0],candidate=scores[2],delta=scores[2]-scores[0]))
        item=dict(group=gi,shape=list(w.shape),timing=timing,cases=cases,
                  cleanup_bitexact=True,delta=sum(c['delta'] for c in cases),
                  adopted=not tree_equal(states[1]['weight_params'],states[2]['weight_params']))
        report['groups'].append(item)
        (ROOT/args.output).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(item),flush=True)
    report.update(complete=True,delta=sum(g['delta'] for g in report['groups']))
    report['total_seconds']={name:sum(sum(g['timing'][name].values()) for g in report['groups'])
                             for name in ('parent','cleanup','joint')}
    (ROOT/args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('delta','total_seconds','validation_checks')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups',default='all')
    p.add_argument('--full-shape',action='store_true')
    p.add_argument('--wide-stress',action='store_true')
    p.add_argument('--output',default='docs/v317-linear-paired.json')
    main(p.parse_args())
