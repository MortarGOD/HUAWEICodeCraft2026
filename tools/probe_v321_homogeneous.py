"""Paired fixed-width QK embedding research."""
from probe_v319_value import ROOT,harness,load_solution


def configured():
    m=load_solution(ROOT/'solution/solution_v320.py')
    exec(compile((ROOT/'tools/attention_homogeneous_core.py').read_text(),'homogeneous-qk','exec'),m.__dict__)
    m._strata_parent_calibration=m._hc_parent_calibration;m.STRATA_DEBUG=m.HC_DEBUG
    def guard(samples,parent,qh,kh,d):
        m._HC_RANKS=(0,1,4) if m._STRATA_BINS==10 else (m._STRATA_BINS,)
        return m._hc_guard(samples,parent,qh,kh,d)
    m._strata_guard=guard
    return m


if __name__=='__main__':
    harness.configured=configured
    harness.main()
