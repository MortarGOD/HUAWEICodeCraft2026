"""Standalone leading-position joint-risk candidates."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def candidate_source(version):
    assert version in (348,349)
    parent=(ROOT/'solution/solution_v324.py').read_text()
    doc=ast.get_source_segment(parent,ast.parse(parent).body[0])
    anchors=1 if version==348 else 4
    parent=parent.replace(doc,f'"""v{version}: {anchors} explicit positions and background joint risk; Linear/Q/K v324."""',1)
    core=(ROOT/'tools/attention_position_risk_core.py').read_text().replace('_PR_ANCHORS=1',f'_PR_ANCHORS={anchors}',1)
    return parent+'\n\n'+core


if __name__=='__main__':
    for version in (348,349):
        source=candidate_source(version);compile(source,str(version),'exec')
        (ROOT/f'solution/solution_v{version}.py').write_text(source);print(version,len(source.splitlines()))
