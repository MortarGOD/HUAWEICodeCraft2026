"""Bounded long-sequence coverage for the already validated Key gauge."""
import ast
from pathlib import Path
from probe_v323_long_key import candidate_source as dynamic_source
ROOT=Path(__file__).resolve().parents[1]


def replace_function(source,name,old,new):
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    body=ast.get_source_segment(source,node)
    assert body.count(old)==1,(name,old)
    changed=body.replace(old,new)
    start=sum(len(line) for line in source.splitlines(keepends=True)[:node.lineno-1])
    end=start+len(body)
    return source[:start]+changed+source[end:]


def candidate_source():
    source=dynamic_source(3)
    long_check="(rows>4096 and (hold['k'][0].numel()>8388608 or not parent['q_state'].get('gq') or 'u' not in parent['q_state']))"
    source=replace_function(source,'_kt_guard','if not 4<=rows<=4096:return parent',
                            'if not 4<=rows<=16384 or '+long_check+':return parent')
    long_check="(t > 4096 and (hold['k'][0].numel() > 8388608 or not parent['q_state'].get('gq') or 'u' not in parent['q_state']))"
    source=replace_function(source,'_ts_guard',"if not 4 <= t <= 4096 or not st.get('gq') or 'u' not in st:",
                            "if not 4 <= t <= 16384 or "+long_check+" or not st.get('gq') or 'u' not in st:")
    end=source.index('"""',source.index('"""')+3)+3
    header='''"""v323: Bounded long-Key Softmax-gauge coverage.
Preserve v322 when calibration and test sequences have <=4096 rows.
Enable calibrated Key translation through
16384 rows and 8388608 Key elements; use 3 offset steps in the long path.
Linear, Query and Value implementations are unchanged. No runtime file I/O.
"""'''
    source=header+source[end:]
    compile(source,'v323','exec');return source


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args()
    source=candidate_source()
    if args.write:
        for name in ('solution_v323.py','solution.py'):(ROOT/'solution'/name).write_text(source)
    print('lines',len(source.splitlines()))
