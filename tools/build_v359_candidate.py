"""Build output-equivalent one-pass calibration variant of v358."""
import ast
from pathlib import Path

from attention_metric_spectral_fast_source import fast_metric_spectral_source

ROOT = Path(__file__).resolve().parents[1]


def candidate_source():
    parent = (ROOT / 'solution/solution_v353.py').read_text()
    doc = ast.get_source_segment(parent, ast.parse(parent).body[0])
    parent = parent.replace(
        doc,
        '"""v359: output-equivalent v358 risk with one-pass calibration Q/K reuse."""',
        1,
    )
    return parent + '\n\n' + fast_metric_spectral_source()


if __name__ == '__main__':
    source = candidate_source()
    compile(source, 'solution_v359.py', 'exec')
    (ROOT / 'solution/solution_v359.py').write_text(source)
    print(len(source.splitlines()))
