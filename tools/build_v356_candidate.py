"""Build standalone v356 from stable v353 plus hybrid positional risk."""
import ast
from pathlib import Path

from attention_hybrid_risk_source import hybrid_risk_source

ROOT = Path(__file__).resolve().parents[1]


def candidate_source():
    parent = (ROOT / 'solution/solution_v353.py').read_text()
    tree = ast.parse(parent)
    doc = ast.get_source_segment(parent, tree.body[0])
    parent = parent.replace(
        doc,
        '"""v356: v353 adaptive V risk plus 5% rank-4 positional spectrum; Linear/Q/K frozen."""',
        1,
    )
    return parent + '\n\n' + hybrid_risk_source(.05, 64, 4, 16)


if __name__ == '__main__':
    source = candidate_source()
    compile(source, 'solution_v356.py', 'exec')
    (ROOT / 'solution/solution_v356.py').write_text(source)
    print(len(source.splitlines()))
