"""Build standalone v358 from v353 plus PSD factor metric and positional spectrum."""
import ast
from pathlib import Path

from attention_metric_spectral_source import metric_spectral_source

ROOT = Path(__file__).resolve().parents[1]


def candidate_source():
    parent = (ROOT / 'solution/solution_v353.py').read_text()
    doc = ast.get_source_segment(parent, ast.parse(parent).body[0])
    parent = parent.replace(
        doc,
        '"""v358: v353 V risk with a learned factor metric and 5% rank-4 positional spectrum."""',
        1,
    )
    return parent + '\n\n' + metric_spectral_source()


if __name__ == '__main__':
    source = candidate_source()
    compile(source, 'solution_v358.py', 'exec')
    (ROOT / 'solution/solution_v358.py').write_text(source)
    print(len(source.splitlines()))
