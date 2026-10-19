"""Paired v318 baseline and independently derived asymmetric experiments."""
import argparse
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import time
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import load_solution, evaluate_attention
from self_check import validate_frozen_state


def candidate_source():
    source = (ROOT/'solution/solution_v318.py').read_text()
    for name in ('_dyn_qk', '_joint_quant_queries'):
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
        old = ast.get_source_segment(source, node)
        if name == '_dyn_qk':
            extra = "    if isinstance(state, dict) and 'aq_p' in state:\n        return _aq_quant(x, state, num_heads, head_dim, x.shape[0])\n"
        else:
            extra = "    if 'aq_p' in state:\n        return _deq_params(_aq_quant(x, state, qh, d, hold['q'][0].shape[0]))\n"
        needle = '    p = _dyn_table(x, None, has_scale=False)'
        assert old.count(needle) == 1
        source = source.replace(old, old.replace(needle, extra+needle))
    return source+'\n\n'+(ROOT/'tools/attention_gptaq_core.py').read_text()


def configured():
    module = load_solution(ROOT/'solution/solution_v318.py')
    exec(compile(candidate_source(), 'v319-gptaq-research', 'exec'), module.__dict__)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups', default='1,5,14,19,47,49')
    p.add_argument('--modes', default='metric,triangular,center')
    p.add_argument('--alpha', type=float, default=0.25)
    p.add_argument('--output', default='docs/v319-gptaq-screen.json')
    a = p.parse_args()
    torch.set_num_threads(4)
    bank = torch.load(ROOT/a.dataset/'attn.pt', weights_only=False)
    indices = list(range(len(bank))) if a.groups=='all' else list(map(int, a.groups.split(',')))
    m = configured(); m._AQ_ALPHA = a.alpha
    report = dict(dataset=a.dataset, groups=indices, alpha=a.alpha, results=[], complete=False,
                  warning='Checkpoint projections with modeled hidden states; no validated online prediction.')
    for gi in indices:
        g = bank[gi]; args = [g[k] for k in ('q_num_heads', 'kv_num_heads', 'head_dim')]
        begin = time.perf_counter(); parent = m._aq_parent_calibration(g['calib'], *args)
        item = dict(group=gi, parent_calibration_seconds=time.perf_counter()-begin, modes={})
        variants = [('parent', parent)]
        for mode in a.modes.split(','):
            m._AQ_MODE = mode
            begin = time.perf_counter()
            proposed = m._aq_fit(g['calib'], parent, *args)
            candidate = parent if proposed is None else m._aq_guard(g['calib'][-1], parent, proposed, *args)
            assert not validate_frozen_state(candidate, mode)
            item['modes'][mode] = dict(extra_calibration_seconds=time.perf_counter()-begin, diagnostics=copy.deepcopy(m.AQ_DEBUG))
            variants.append((mode, candidate))
        for name, st in variants:
            m.hif4_calibration_attention = lambda *unused, _st=st: copy.deepcopy(_st)
            with contextlib.redirect_stdout(io.StringIO()):
                scores, seconds = evaluate_attention(m, [g])
            item['modes'].setdefault(name, {}).update(score=sum(scores), cases=scores, seconds=seconds)
        for mode in a.modes.split(','):
            item['modes'][mode]['delta'] = item['modes'][mode]['score']-item['modes']['parent']['score']
        report['results'].append(item)
        (ROOT/a.output).write_text(json.dumps(report, indent=2)+'\n')
        print(gi, {name: data.get('delta', 0) for name, data in item['modes'].items()}, flush=True)
    report.update(complete=True, totals={mode: sum(r['modes'][mode]['delta'] for r in report['results']) for mode in a.modes.split(',')})
    (ROOT/a.output).write_text(json.dumps(report, indent=2)+'\n')
    print(report['totals'], flush=True)


if __name__ == '__main__':
    main()
