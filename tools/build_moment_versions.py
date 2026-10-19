"""Standalone conditional second-moment and cross-gradient probes."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def candidate_source(version):
    assert version in (341,342,350,351)
    parent=(ROOT/'solution/solution_v324.py').read_text()
    doc=ast.get_source_segment(parent,ast.parse(parent).body[0])
    parent=parent.replace(doc,f'"""v{version}: conditional Value moment search; Linear/Q/K v324."""',1)
    core=(ROOT/'tools/attention_moment_core.py').read_text()
    if version==342:core=core.replace('_AM_GRADIENT_GAIN=0.','_AM_GRADIENT_GAIN=.25',1)
    if version in (350,351):
        core+='\n\n'+(ROOT/'tools/attention_factor_risk_core.py').read_text().replace('_AF_SEARCH_RIDGE=.1','_AF_SEARCH_RIDGE=1.',1)
    if version==351:
        core+='\n\n'+(ROOT/'tools/attention_factor_fast_guard.py').read_text()
        core+='\n\n'+(ROOT/'tools/attention_factor_cal_cache.py').read_text()
    return parent+'\n\n'+core


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--factor',action='store_true');parser.add_argument('--cal-cache',action='store_true');args=parser.parse_args()
    for version in ((351,) if args.cal_cache else ((350,) if args.factor else (341,342))):
        source=candidate_source(version);compile(source,str(version),'exec')
        (ROOT/f'solution/solution_v{version}.py').write_text(source)
        print(version,len(source.splitlines()))
