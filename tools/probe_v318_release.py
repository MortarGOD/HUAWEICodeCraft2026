"""Conservative release: full-query tiled calibration, long Grams, signed GPTQ scratch."""
import ast
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from probe_v318_bounded import candidate_source as bounded_source


def candidate_source():
    original=(ROOT/'solution/solution_v317.py').read_text()
    source=bounded_source()
    original_nodes=ast.parse(original).body
    base=next(n for n in original_nodes if isinstance(n,ast.FunctionDef) and n.name=='hif4_calibration_attention')
    old_function=ast.get_source_segment(original,base)
    start=old_function.index('    # ---- guard setup: hold sample quantized ONCE ----')
    end=old_function.index('    q_state = {"rot": rot, "rseed": rseed, "kvh": kvh}',start)
    old_guard=old_function[start:end]
    start=source.index('    # Bounded independent queries, all keys, actual quantized V.')
    end=source.index('    q_state = {"rot": rot, "rseed": rseed, "kvh": kvh}',start)
    source=source[:start]+old_guard+source[end:]
    # No unused experimental helper or parameter reaches the submission.
    source=source[:source.index('\n\n"""Bounded-query full-key deployment guard;')]
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_attention_out')
    old=ast.get_source_segment(source,node)
    new='''def _attention_out(q, k, v, qh, kvh, dh):
    seq = q.shape[0]
    qf = q.view(seq, qh, dh).transpose(0, 1)
    kf = k.view(seq, kvh, dh).transpose(0, 1).repeat_interleave(qh // kvh, dim=0)
    vf = v.view(seq, kvh, dh).transpose(0, 1).repeat_interleave(qh // kvh, dim=0)
    if seq <= 512:
        scores = torch.bmm(qf, kf.transpose(1, 2)) / (dh ** 0.5)
        prob = torch.softmax(scores, dim=-1)
        out = torch.bmm(prob, vf)
    else:
        # Exact full-key softmax for EVERY query, tiled only along independent
        # query rows. No key/query sampling and no changed guard objective.
        chunk = max(1, min(256, 4_000_000 // max(1, qh * seq)))
        out = torch.empty(qh, seq, dh, dtype=qf.dtype, device=qf.device)
        kt = kf.transpose(1, 2)
        for start in range(0, seq, chunk):
            stop = min(start + chunk, seq)
            scores = torch.bmm(qf[:, start:stop], kt) / (dh ** 0.5)
            prob = torch.softmax(scores, dim=-1)
            out[:, start:stop] = torch.bmm(prob, vf)
    return out.transpose(0, 1).reshape(seq, qh * dh)'''
    source=source.replace(old,new)
    start=source.index('"""');end=source.index('"""',start+3)+3
    header='''"""v318: tiled full-query calibration and signed GPTQ scratch.

Linear is unchanged from v317. Preserve Attention guard objectives and all
queries/keys; include bounded statistics from long calibration samples.
"""'''
    return header+source[end:]


def configured():
    m=load_solution(ROOT/'solution/solution_v317.py')
    exec(compile(candidate_source(),'v318-release','exec'),m.__dict__)
    return m
