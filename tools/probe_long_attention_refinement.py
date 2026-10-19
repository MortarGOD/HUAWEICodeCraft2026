"""Offline probe of the hard long-sequence Q/K refinement cutoff."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import types

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import evaluate_attention
from probe_persistent_covariance import load


def candidate_source(rounds=4, cap=4096, temperature=False):
    if temperature:
        from probe_attention_temperature import candidate_source as temperature_source
        source = temperature_source() + '\n_TEMP_CENTER_KEYS = False\n'
    else:
        source = (ROOT/'solution/solution_v314.py').read_text()
    old = 'ATTN_REFINE_MAX_T = 1024'
    assert source.count(old) == 1
    source = source.replace(old, f'ATTN_REFINE_MAX_T = {cap}')
    old = 'int(state["rf"]))\n                values = v1.contiguous()'
    assert source.count(old) == 1
    source = source.replace(old, f'int(state["rf"]), total_rounds=(int(state["rf"])*{rounds} if x.shape[0]>1024 else None))\n                values = v1.contiguous()')
    old = 'rounds = rf * (32 if rows <= 128 else _REF_ATTN_ROUNDS)'
    assert source.count(old) == 1
    source = source.replace(old, f'rounds = rf * (32 if rows <= 128 else ({rounds} if rows>1024 else _REF_ATTN_ROUNDS))')
    return source


def configured(rounds, temperature=False):
    if not rounds:
        return load(314)
    m = types.ModuleType('long_attention')
    exec(compile(candidate_source(rounds, temperature=temperature), 'long_attention', 'exec'), m.__dict__)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups', default='1,5,14,26,37,49')
    p.add_argument('--rounds', default='0,4,8')
    p.add_argument('--temperature', action='store_true')
    p.add_argument('--output', default='docs/v315-long-refinement-screen.json')
    a = p.parse_args()
    bank = torch.load(ROOT/a.dataset/'attn.pt', weights_only=False)
    groups = bank if a.groups == 'all' else [bank[int(i)] for i in a.groups.split(',')]
    result = dict(dataset=a.dataset, groups=a.groups, temperature=a.temperature, results={},
                  warning='Diagnostic modeled activations; not a qualified online prediction.')
    for rounds in map(int, a.rounds.split(',')):
        m = configured(rounds, a.temperature)
        with contextlib.redirect_stdout(io.StringIO()):
            scores, seconds = evaluate_attention(m, groups)
        result['results'][str(rounds)] = dict(score=sum(scores), seconds=seconds, cases=scores)
        (ROOT/a.output).write_text(json.dumps(result, indent=2)+'\n')
        print(rounds, sum(scores), seconds, flush=True)


if __name__ == '__main__':
    main()
