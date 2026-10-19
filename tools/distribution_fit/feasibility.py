"""Row-scaled ADMM feasibility solver for a shared simplex population."""
import numpy as np
from tools.distribution_fit.fit import simplex

def solve_feasible(A,low,high,initial=None,max_iterations=50000):
 scale=np.maximum(np.linalg.norm(A,axis=1),1e-30)
 B=A/scale[:,None];lo=low/scale;hi=high/scale
 n=A.shape[1];u=np.full(n,1/n) if initial is None else simplex(initial)
 z=np.clip(B@u,lo,hi);dual_z=np.zeros(len(lo));dual_u=np.zeros(n)
 inverse=np.linalg.inv(B.T@B+np.eye(n))
 best=u.copy();best_violation=float('inf')
 tolerance=np.maximum((high-low)/2,1e-12)
 for iteration in range(max_iterations):
  x=inverse@(B.T@(z-dual_z)+u-dual_u)
  bx=B@x
  z=np.clip(bx+dual_z,lo,hi);u=simplex(x+dual_u)
  dual_z+=bx-z;dual_u+=x-u
  if iteration%10==0:
   value=A@u
   violation=float(np.max(np.maximum(low-value,value-high)/tolerance))
   if violation<best_violation:best=u.copy();best_violation=violation
   if violation<=1e-8:break
 return best,{'method':'row-scaled ADMM feasibility','iterations':iteration+1,'maximum_normalized_interval_violation':best_violation,'limitations':'Failure to converge is not an infeasibility certificate.'}
