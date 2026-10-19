"""Meaningful numerical checks: known feasible mixture, direction, and impossible support."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.distribution_fit.fit import solve,simplex
A=np.array([[1.,-2.,3.],[-4.,2.,1.],[2.,0.,-1.],[0.,3.,-2.]])*100
truth=np.array([.2,.3,.5]);y=A@truth;tol=np.maximum(1,np.abs(y)*.05)
w,it=solve(A,y,tol,y-tol,y+tol)
assert np.all(w>=0) and abs(w.sum()-1)<1e-12
assert np.all(np.abs(A@w-y)<=tol+1e-6),(w,A@w-y)
assert np.all(np.sign(A@w)==np.sign(y))
B=np.array([[1.,2.,3.]])
w,_=solve(B,np.array([-10.]),np.ones(1),np.array([-11.]),np.array([-9.]))
assert (B@w)[0]>=1 and abs((B@w)[0]+10)>1
assert np.allclose(simplex(np.array([-.2,.5,1.])),[0,.25,.75])
print('known feasible interval, direction, simplex and impossible-support checks passed')

from tools.distribution_fit.feasibility import solve_feasible
A=np.array([[1e8,0.,0.],[0.,1.,-1.]])
y=np.array([0.,.5]);tol=np.array([1.,.01])
w,info=solve_feasible(A,y-tol,y+tol)
assert np.all(np.abs(A@w-y)<=tol+1e-7),(w,info)
assert np.all(w>=0) and abs(w.sum()-1)<1e-12
print('ill-conditioned known-feasible ADMM check passed')

from tools.distribution_fit.fit import separation_certificate
A=np.array([[0.,1.],[1.,0.]])
c=separation_certificate(A,np.array([.8,.8]),np.array([.9,.9]),np.array([.5,.5]))
assert c['certified'] and c['gap']>0
c=separation_certificate(A,np.array([.4,.4]),np.array([.6,.6]),np.array([.5,.5]))
assert not c['certified']
print('joint support separation and feasible non-certificate checks passed')
