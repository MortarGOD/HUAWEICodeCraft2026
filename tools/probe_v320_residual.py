"""Paired residual-risk research, parent calibration shared once per group."""
from probe_v319_value import ROOT,harness,load_solution


def configured():
    m=load_solution(ROOT/'solution/solution_v319.py')
    exec(compile((ROOT/'tools/attention_residual_core.py').read_text(),'residual-risk','exec'),m.__dict__)
    m._strata_parent_calibration=m._vr_parent_calibration
    m.STRATA_DEBUG=m.VR_DEBUG
    def guard(samples,parent,qh,kh,d):
        m._VR_TARGET={0:'cross',1:'full',2:'zero',3:'cross'}[m._STRATA_BINS]
        m._VR_FIT_SEQUENCES=2 if m._STRATA_BINS==3 else 4
        return m._vr_guard(samples,parent,qh,kh,d)
    m._strata_guard=guard
    return m


if __name__=='__main__':
    harness.configured=configured
    harness.main()
