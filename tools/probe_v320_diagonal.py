"""Same target/guard, constrain learned cross-channel matrix to a diagonal+bias."""
import ast
from probe_v319_value import ROOT,harness,load_solution


def configured():
    m=load_solution(ROOT/'solution/solution_v320.py')
    full=m._vr_fit
    source=(ROOT/'solution/solution_v320.py').read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_vr_fit')
    fit=ast.get_source_segment(source,node)
    old="    st['beta']=torch.linalg.solve(aa+reg[:,None,None]*torch.eye(d+1),ab).contiguous()"
    replacement="""    idx=torch.arange(d)
    h00=aa[:,idx,idx]+reg[:,None]
    h01=aa[:,idx,-1]
    h11=aa[:,-1,-1][:,None]+reg[:,None]
    b0=ab[:,idx,idx];b1=ab[:,-1,:]
    determinant=(h00*h11-h01.square()).clamp_min(1e-30)
    beta=torch.zeros(kh,d+1,d)
    beta[:,idx,idx]=(h11*b0-h01*b1)/determinant
    beta[:,-1,:]=(h00*b1-h01*b0)/determinant
    st['beta']=beta.contiguous()"""
    assert old in fit
    exec(compile(fit.replace(old,replacement),'diagonal-normal-equations','exec'),m.__dict__)
    diagonal=m._vr_fit;m._vr_fit=full
    m._strata_parent_calibration=m._vr_parent_calibration
    m.STRATA_DEBUG=m.VR_DEBUG
    def guard(samples,parent,qh,kh,d):
        m._vr_fit=full if m._STRATA_BINS==0 else diagonal
        st=m._vr_guard(samples,parent,qh,kh,d)
        return {k:v for k,v in st.items() if k!='_vr_cache'}
    m._strata_guard=guard
    return m


if __name__=='__main__':
    harness.configured=configured
    harness.main()
