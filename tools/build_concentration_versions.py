"""Generate static standalone concentration-risk candidates; offline AST extraction."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def candidate_source(version):
    assert version in (345,346)
    source=(ROOT/'solution/solution_v343.py').read_text()
    tree=ast.parse(source);doc=ast.get_source_segment(source,tree.body[0])
    guard=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_ms_guard')
    evaluate=ast.get_source_segment(source,guard).replace('def _ms_guard(samples,parent,qh,kh,d):','def _cr_evaluate(samples,parent,qh,kh,d,penalty):',1)
    evaluate=evaluate.replace('penalty=_MS_PENALTY,pairs=_MS_PAIRS','penalty=penalty,pairs=_MS_PAIRS',1)
    research=(ROOT/'tools/probe_concentration_risk.py').read_text()
    estimator=next(n for n in ast.parse(research).body if isinstance(n,ast.FunctionDef) and n.name=='estimate')
    estimate=ast.get_source_segment(research,estimator).replace('def estimate(m,','def _cr_estimate(',1).replace('m.','')
    strength='1.' if version==345 else '.1'
    tail=f'''\n\ndef _ms_guard(samples,parent,qh,kh,d):
    MS_DEBUG.clear();MS_DEBUG.update(error=False,adopted=0)
    if len(samples)<3 or d%64 or not parent['q_state'].get('gq'):return parent
    hold=samples[-1]
    if not 4<=hold['v'][0].shape[0]<=4096 or hold['v'][0].numel()>4194304:return parent
    penalty,statistics=_cr_estimate(samples,parent,qh,kh,d)
    if penalty is None:return parent
    penalty=penalty*{strength}
    result=_cr_evaluate(samples,parent,qh,kh,d,penalty)
    MS_DEBUG.update(penalty=penalty.flatten().tolist(),concentration=statistics)
    return result
'''
    source=source.replace(doc,f'"""v{version}: calibrated concentration-risk Value search; Linear/Q/K v324."""',1)
    return source+'\n\n'+estimate+'\n\n'+evaluate+tail


if __name__=='__main__':
    for v in (345,346):
        source=candidate_source(v);compile(source,str(v),'exec')
        (ROOT/f'solution/solution_v{v}.py').write_text(source);print(v,len(source.splitlines()))
