"""Build static single-file Query scale-tree candidate, Linear frozen v324."""
import argparse,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def candidate_source():
    parent=(ROOT/'solution/solution_v324.py').read_text()
    header=ast.get_source_segment(parent,ast.parse(parent).body[0])
    parent=parent.replace(header,'"""v325: Query scale-tree block reconstruction after GPTQ.\nLinear, Key and Value retain v324; no runtime file access.\n"""',1)
    core=(ROOT/'tools/attention_query_grid_core.py').read_text()
    guard=next(n for n in ast.parse(core).body if isinstance(n,ast.FunctionDef) and n.name=='_qg_guard')
    tail=(ROOT/'tools/attention_query_grid_release.py').read_text()
    return parent+'\n\n'+tail+'\n\n'+ast.get_source_segment(core,guard)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
    source=candidate_source();compile(source,'v325','exec')
    if a.write:
        for name in ('solution_v325.py','solution.py'):(ROOT/'solution'/name).write_text(source)
    print('lines',len(source.splitlines()))
