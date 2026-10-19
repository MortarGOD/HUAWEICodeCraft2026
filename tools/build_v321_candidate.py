"""Mechanical single-file candidate construction; never imported by submission."""
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[1]


def candidate_source():
    source=(ROOT/'solution/solution_v320.py').read_text()
    end=source.index('"""',source.index('"""')+3)+3
    header='''"""v321: Softmax-invariant current-Key offset optimization.

Calibration gates exact deployed PV error. Independent dynamic Key performs
three cheap lattice-offset updates before one GPTQ pass on the shifted input.
Linear, Query and Value implementations are unchanged from v320.
"""'''
    helpers=(ROOT/'tools/attention_homogeneous_core.py').read_text()
    functions=[ast.get_source_segment(helpers,n) for n in ast.parse(helpers).body
               if isinstance(n,ast.FunctionDef) and n.name in ('_hc_parent_input','_hc_quantize_raw')]
    helpers='\n\n'.join(functions).replace('_hc_parent_input','_kt_input').replace('_hc_quantize_raw','_kt_fixed_grid').replace('_hc_encode','_kt_input')
    core=(ROOT/'tools/attention_key_translation_core.py').read_text()
    core=core.replace('_KT_MODE=1','_KT_MODE=2').replace('_KT_MODES=None','_KT_MODES=(2,4)').replace('_kt_parent_calibration=_hc_parent_calibration','_kt_parent_calibration=hif4_calibration_attention')
    core=core.replace('_KT_BUDGET_ROUTE=False','_KT_BUDGET_ROUTE=True')
    core=core.replace('_KT_GUARD_ROWS=64','_KT_GUARD_ROWS=32').replace('_KT_MIN_GAIN=.005','_KT_MIN_GAIN=.01')
    core=core.replace('_kt_parent_k=_hc_parent_k','_kt_parent_k=hif4_dynamic_quantize_k').replace('hif4_dynamic_quantize_q=_hc_parent_q\n','')
    core=core.replace('_hc_parent_input','_kt_input').replace('_hc_quantize_raw','_kt_fixed_grid')
    return header+source[end:]+'\n\n'+helpers+'\n\n'+core


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args()
    source=candidate_source();compile(source,'v321','exec')
    if args.write:
        for name in ('solution_v321.py','solution.py'):(ROOT/'solution'/name).write_text(source)
    print('lines',len(source.splitlines()))
