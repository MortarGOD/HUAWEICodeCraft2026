"""Reuse fixed-parent deployment guard to compare bounded beam candidates."""
import ast
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
import probe_v318_metadata as harness


def candidate_source():
    source=(ROOT/'solution/solution_v318.py').read_text()
    for name in ('_dyn_qk', '_joint_quant_queries'):
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
        old=ast.get_source_segment(source,node)
        if name=='_dyn_qk':
            tail='    return _values_to_params(values, p) if values is not None else p'
            new='    p = _values_to_params(values, p) if values is not None else p\n    return _beam_refine(x, p, state, num_heads, head_dim, x.shape[0])'
        else:
            tail='    return _deq_params(_values_to_params(values, p))'
            new='    p = _values_to_params(values, p)\n    return _deq_params(_beam_refine(x, p, state, qh, d, rows))'
        assert old.count(tail)==1
        source=source.replace(old,old.replace(tail,new))
    metadata=(ROOT/'tools/attention_metadata_core.py').read_text()
    guard=metadata[metadata.index('def _meta_guard'):]
    guard=guard.replace('_meta_prepare','_beam_prepare').replace('meta_active','beam_active')
    return source+'\n\n'+(ROOT/'tools/attention_beam_core.py').read_text()+'''\n
_META_MAX_T = 4096
_META_MARGIN = .005
META_DEBUG = {}
_meta_parent_calibration = hif4_calibration_attention
'''+guard


def configured():
    m=load_solution(ROOT/'solution/solution_v318.py')
    exec(compile(candidate_source(),'beam-research','exec'),m.__dict__)
    return m


if __name__=='__main__':
    harness.configured=configured
    harness.main()
