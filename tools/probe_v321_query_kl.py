"""Pairwise calibration-key Softmax KL research."""
from probe_v319_value import ROOT,harness,load_solution


def configured():
    m=load_solution(ROOT/'solution/solution_v320.py')
    exec(compile((ROOT/'tools/attention_query_kl_core.py').read_text(),'query-kl','exec'),m.__dict__)
    m._strata_parent_calibration=m._nq_parent_calibration;m.STRATA_DEBUG=m.NQ_DEBUG
    def guard(samples,parent,qh,kh,d):
        m._NQ_ROUNDS=m._STRATA_BINS%100;m._NQ_OUTPUT=100<=m._STRATA_BINS<200
        m._NQ_ALIGNED=m._STRATA_BINS>=200
        return m._nq_guard(samples,parent,qh,kh,d)
    m._strata_guard=guard
    return m


if __name__=='__main__':
    harness.configured=configured;harness.main()
