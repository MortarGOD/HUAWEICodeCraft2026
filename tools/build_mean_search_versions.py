"""Independent standalone single-coordinate control and pair-search probe."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def candidate_source(version):
    assert version in (343,344)
    parent=(ROOT/'solution/solution_v324.py').read_text()
    doc=ast.get_source_segment(parent,ast.parse(parent).body[0])
    parent=parent.replace(doc,f'"""v{version}: native Value mean-risk search; Linear/Q/K v324."""',1)
    core=(ROOT/'tools/attention_mean_search_core.py').read_text()
    if version==343:core=core.replace('_MS_PAIRS=2','_MS_PAIRS=0',1)
    return parent+'\n\n'+core


if __name__=='__main__':
    for version in (343,344):
        source=candidate_source(version);compile(source,str(version),'exec')
        (ROOT/f'solution/solution_v{version}.py').write_text(source)
        print(version,len(source.splitlines()))
