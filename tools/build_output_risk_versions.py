"""Static standalone builds for final-output-risk experiments."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def candidate_source(version):
    assert version in (339,340)
    parent=(ROOT/'solution/solution_v324.py').read_text()
    doc=ast.get_source_segment(parent,ast.parse(parent).body[0])
    label='native Value block output-risk ranking' if version==339 else 'nonlinear per-channel Value residual student'
    parent=parent.replace(doc,f'"""v{version}: {label}; Linear/Q/K v324."""',1)
    core='attention_risk_rank_core.py' if version==339 else 'attention_residual_student_core.py'
    tail=(ROOT/'tools'/core).read_text()
    if version==339:tail=tail.replace('_RK_CONTEXT=True','_RK_CONTEXT=False',1)
    return parent+'\n\n'+tail


if __name__=='__main__':
    for version in (339,340):
        source=candidate_source(version);compile(source,f'v{version}','exec')
        (ROOT/f'solution/solution_v{version}.py').write_text(source)
        print(version,len(source.splitlines()))
