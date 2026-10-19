"""Paired tests of statically constructed, routed nearby-scale candidate."""
from probe_v319_value import ROOT,harness,load_solution
from build_v319_candidate import candidate_source


def configured():
    m=load_solution(ROOT/'solution/solution_v318.py')
    exec(compile(candidate_source(),'v319-release-research','exec'),m.__dict__)
    m._strata_parent_calibration=m._vm_parent_calibration
    m.STRATA_DEBUG=m.VM_DEBUG
    m._value_calibration=m.hif4_calibration_attention
    def guard(samples,parent,qh,kh,d):
        previous=m._vm_parent_calibration
        m._vm_parent_calibration=lambda *unused:parent
        try:return m._value_calibration(samples,qh,kh,d)
        finally:m._vm_parent_calibration=previous
    m._strata_guard=guard
    return m


if __name__=='__main__':
    harness.configured=configured
    harness.main()
