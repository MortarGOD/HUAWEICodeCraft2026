"""Research-only coupled eight-coordinate directions on native four-code scales."""
import argparse
import ast
import copy
import json
import torch
from probe_v319_value import ROOT, load_solution
from hif4_quant import dequantize_hif4, quantize_to_hif4
from self_check import validate_hif4_params


def install(m, correlated):
    source = (ROOT/'solution/solution_v324.py').read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == '_quantize_linear_exact_target')
    body = ast.get_source_segment(source, node)
    point = '    result_lv3 = result["scale_lv3"].reshape('
    setup = '''    permutations = []
    pair_inverses = []
    for bi in range(block_count):
        if PAIR_CORRELATED:
            affinity = block_gram[bi].reshape(16,4,16,4).square().sum((1,3))
            norm = affinity.diagonal().clamp_min(1e-20).sqrt()
            affinity = affinity / (norm[:,None] * norm[None,:])
            available = list(range(16))
            groups = []
            while available:
                a = available.pop(0)
                b = max(available, key=lambda j: float(affinity[a,j]))
                available.remove(b)
                groups.extend((a,b))
            perm = torch.tensor([g*4+c for g in groups for c in range(4)])
        else:
            perm = torch.arange(64)
        g = block_gram[bi][perm][:,perm].reshape(8,8,8,8)
        h = torch.stack([g[j,:,j,:] for j in range(8)])
        reg = .01*h.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-20)
        pair_inverses.append(torch.linalg.inv(h+reg[:,None,None]*torch.eye(8)))
        permutations.append(perm)
'''
    assert body.count(point) == 1
    body = body.replace(point, setup+point)
    old = '''            direction = torch.einsum(
                "rmc,mcd->rmd", gradient, micro_inverse[block]
            )'''
    new = '''            perm = permutations[block]
            paired = gradient_flat[:,perm].reshape(flat_rows,8,8)
            solved = torch.einsum('rmc,mcd->rmd',paired,pair_inverses[block]).reshape(flat_rows,64)
            direction_flat = torch.empty_like(gradient_flat)
            direction_flat[:,perm] = solved
            direction = direction_flat.reshape(flat_rows,16,4)'''
    assert body.count(old) == 1
    body = body.replace(old,new)
    m.PAIR_CORRELATED = correlated
    exec(compile(body, 'v325_pair_research', 'exec'), m.__dict__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='local_fitted_models_attention_feedback_v2')
    parser.add_argument('--output', default='docs/v325-pairs-screen.json')
    args = parser.parse_args()
    torch.set_num_threads(4)
    parent = load_solution(ROOT/'solution/solution_v324.py')
    modules = {}
    for name, correlated in [('adjacent',False),('correlated',True)]:
        module=load_solution(ROOT/'solution/solution_v324.py')
        install(module,correlated)
        modules[name]=module
    bank=torch.load(ROOT/args.dataset/'linear.pt',weights_only=False)
    records=[]
    for gi in (0,4,6,7,9):
        g=bank[gi]
        calibration=parent.hif4_calibration_and_quantize_weight(*g['weight'],g['calib_activation_list'])
        state=calibration['activation_state']
        saved=copy.deepcopy(state)
        w=parent.dequantize_nvfp4(*g['weight']).float()
        b=parent._linear_score_decode(calibration['weight_params']).float()
        bw=dequantize_hif4(quantize_to_hif4(w),tuple(w.shape)).float()
        for ti,packed in enumerate(g['test_activation_list']):
            x=parent.dequantize_nvfp4(*packed).float(); y=x@w.T
            xb=dequantize_hif4(quantize_to_hif4(x),tuple(x.shape)).float()
            baseline=float((y-xb@bw.T).square().mean())
            scores={}
            for name,module in [('parent',parent),*modules.items()]:
                params=module.hif4_dynamic_quantize_activation(*packed,state)
                assert not validate_hif4_params(params,packed[0].shape,name)
                z=parent._linear_score_decode(params).float()
                scores[name]=1-float((y-z@b.T).square().mean())/baseline
            records.append(dict(group=gi,test=ti,scores=scores))
        assert all(torch.equal(v,saved[k]) for k,v in state.items() if isinstance(v,torch.Tensor))
        print(gi,{name:sum(r['scores'][name]-r['scores']['parent'] for r in records) for name in modules},flush=True)
    (ROOT/args.output).write_text(json.dumps(dict(complete=True,dataset=args.dataset,rows=records,delta={name:sum(r['scores'][name]-r['scores']['parent'] for r in records) for name in modules},score_prediction_allowed=False),indent=2)+'\n')


if __name__=='__main__':main()
