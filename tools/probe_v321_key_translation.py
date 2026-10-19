"""Paired dynamic key translation research."""
from probe_v319_value import ROOT,harness,load_solution


def configured():
    m=load_solution(ROOT/'solution/solution_v320.py')
    for file in ('attention_homogeneous_core.py','attention_key_translation_core.py'):
        exec(compile((ROOT/'tools'/file).read_text(),file,'exec'),m.__dict__)
    m._strata_parent_calibration=m._kt_parent_calibration;m.STRATA_DEBUG=m.KT_DEBUG
    def guard(samples,parent,qh,kh,d):
        m._KT_MODE=m._STRATA_BINS;m._KT_MODES=(2,4) if m._STRATA_BINS==24 else None
        return m._kt_guard(samples,parent,qh,kh,d)
    m._strata_guard=guard
    return m


if __name__=='__main__':
    harness.configured=configured;harness.main()
