"""Channel-layout ablation on identical parent calibration states."""
import probe_v323_sensitivity as harness
from probe_v319_value import ROOT,load_solution


def configured():
    m=load_solution(ROOT/'solution/solution_v322.py')
    for file in ('attention_centered_balance_core.py','attention_channel_layout_core.py'):
        exec(compile((ROOT/'tools'/file).read_text(),file,'exec'),m.__dict__)
    m._ss_parent_calibration=m._sb_parent_calibration;m.SS_DEBUG=m.SB_DEBUG
    def guard(*args):
        m._SP_MODE=m._SS_MODE
        return m._sb_guard(*args)
    m._ss_guard=guard
    return m


if __name__=='__main__':
    harness.configured=configured
    harness.main()
