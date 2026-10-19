"""Build and evaluate bounded-calibration candidate; only local research uses exec."""
import argparse
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution,evaluate_attention


def candidate_source():
    source=(ROOT/'solution/solution_v317.py').read_text()
    scratch=(ROOT/'tools/gptq_scratch_core.py').read_text()
    replacements={n.name:ast.get_source_segment(scratch,n) for n in ast.parse(scratch).body if isinstance(n,ast.FunctionDef)}
    tree=ast.parse(source)
    for node in reversed(tree.body):
        if isinstance(node,ast.FunctionDef) and node.name in replacements:
            source=source.replace(ast.get_source_segment((ROOT/'solution/solution_v317.py').read_text(),node),replacements[node.name])
    old='''            if Tt > 1024:
                continue
            qd = dequantize_nvfp4(*smp["q"]).float()
            kd = dequantize_nvfp4(*smp["k"]).float()'''
    new='''            gram_weight = 1.0
            if Tt > 1024:
                selected = torch.linspace(0, Tt - 1, 1024).long()
                qd = dequantize_nvfp4(smp["q"][0][selected], smp["q"][1][selected]).float()
                kd = dequantize_nvfp4(smp["k"][0][selected], smp["k"][1][selected]).float()
                gram_weight = Tt / 1024.0
                Tt = 1024
            else:
                qd = dequantize_nvfp4(*smp["q"]).float()
                kd = dequantize_nvfp4(*smp["k"]).float()'''
    assert source.count(old)==1;source=source.replace(old,new)
    old='''            G_q = torch.bmm(qb.transpose(1, 2), qb)
            for hv in range(kvh):'''
    new='''            G_q = torch.bmm(qb.transpose(1, 2), qb)
            if gram_weight != 1.0:
                G_k = G_k * gram_weight
                G_q = G_q * gram_weight
            for hv in range(kvh):'''
    assert source.count(old)==1;source=source.replace(old,new)
    start=source.index('    # ---- guard setup: hold sample quantized ONCE ----')
    end=source.index('    q_state = {"rot": rot, "rseed": rseed, "kvh": kvh}',start)
    source=source[:start]+'''    # Bounded independent queries, all keys, actual quantized V.
    if qk_ready:
        gq, rf_q, rf_k = _cal_qk_guard(qf_full, kf_full, vb_full, rot, R,
                                      Hk, Hq, Uq, Uk, qh, kvh, dh)
        if gq:
            u_q, u_k = Uq.contiguous(), Uk.contiguous()
        Hk_bf = Hk.to(torch.bfloat16).contiguous()
        Hq_bf = Hq.to(torch.bfloat16).contiguous()

'''+source[end:]
    return source+'\n\n'+(ROOT/'tools/attention_bounded_core.py').read_text()


def configured():
    m=load_solution(ROOT/'solution/solution_v317.py')
    exec(compile(candidate_source(),'bounded-research','exec'),m.__dict__)
    return m


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups',default='1,5,14,19,47,49')
    p.add_argument('--output',default='docs/v318-bounded-screen.json')
    a=p.parse_args();torch.set_num_threads(4)
    bank=torch.load(ROOT/a.dataset/'attn.pt',weights_only=False)
    indices=list(range(len(bank))) if a.groups=='all' else list(map(int,a.groups.split(',')))
    report=dict(dataset=a.dataset,groups=indices,results={},warning='Fixed diagnostic bank, modeled activations, not a calibrated online estimate.')
    for name,m in (('parent',load_solution(ROOT/'solution/solution_v317.py')),('candidate',configured())):
        with contextlib.redirect_stdout(io.StringIO()):scores,seconds=evaluate_attention(m,[bank[i] for i in indices])
        report['results'][name]=dict(score=sum(scores),seconds=seconds,cases=scores)
        (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n');print(name,sum(scores),seconds,flush=True)
    report['delta']=report['results']['candidate']['score']-report['results']['parent']['score']
    (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n');print('DELTA',report['delta'],flush=True)


if __name__=='__main__':main()
