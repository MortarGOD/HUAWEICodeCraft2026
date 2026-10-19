"""Build standalone v357 from v353 plus a learned factor-space PSD metric."""
import ast
from pathlib import Path

from attention_factor_metric_source import factor_metric_source

ROOT = Path(__file__).resolve().parents[1]


def candidate_source():
    parent = (ROOT / 'solution/solution_v353.py').read_text()
    doc = ast.get_source_segment(parent, ast.parse(parent).body[0])
    parent = parent.replace(
        doc,
        '"""v357: v353 adaptive V risk with a calibration-learned 16x16 PSD factor metric."""',
        1,
    )
    return parent + '\n\n' + factor_metric_source(.25, .01)


if __name__ == '__main__':
    source = candidate_source()
    compile(source, 'solution_v357.py', 'exec')
    (ROOT / 'solution/solution_v357.py').write_text(source)
    print(len(source.splitlines()))
