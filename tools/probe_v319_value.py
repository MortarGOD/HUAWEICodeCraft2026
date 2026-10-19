"""Full-bank paired validation of the previously bounded Value scale research."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
import probe_v319_strata as harness


def candidate_source():
    source=(ROOT/'solution/solution_v318.py').read_text()
    for name in ('value_metadata_core.py','value_metadata_inner_core.py','value_metadata_row_core.py'):
        source+='\n\n'+(ROOT/'tools'/name).read_text()
    source+='\n_VM_VARIANCE=8.0\n_VM_CAP=1.1\n'
    return source


def configured():
    m=load_solution(ROOT/'solution/solution_v318.py')
    exec(compile(candidate_source(),'value-tree-research','exec'),m.__dict__)
    m._strata_parent_calibration=m._vm_parent_calibration
    m.STRATA_DEBUG=m.VM_DEBUG
    def guard(samples,parent,qh,kh,d):
        old=m._vm_parent_calibration
        m._vm_parent_calibration=lambda *unused:parent
        try:return m._value_calibration(samples,qh,kh,d)
        finally:m._vm_parent_calibration=old
    m._value_calibration=m.hif4_calibration_attention
    m._strata_guard=guard
    return m


if __name__=='__main__':
    harness.configured=configured
    harness.main()
