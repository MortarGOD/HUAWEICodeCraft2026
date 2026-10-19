"""Reuse the fixed-data paired harness for centered reciprocal balancing."""
import probe_v323_sensitivity as harness
from probe_v319_value import ROOT,load_solution


def configured():
    m=load_solution(ROOT/'solution/solution_v322.py')
    exec(compile((ROOT/'tools/attention_centered_balance_core.py').read_text(),'centered-balance','exec'),m.__dict__)
    m._ss_parent_calibration=m._sb_parent_calibration;m.SS_DEBUG=m.SB_DEBUG
    def guard(*args):
        m._SB_ALPHA={1:.25,2:.5,3:1.}[m._SS_MODE]
        return m._sb_guard(*args)
    m._ss_guard=guard
    return m


if __name__=='__main__':
    harness.configured=configured
    harness.main()
