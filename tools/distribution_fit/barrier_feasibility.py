"""NumPy-only phase-I log-barrier search on G w >= h, w in simplex."""
import numpy as np

def phase_one(G,h,initial=None):
    m,n=G.shape
    scale=np.maximum(np.max(np.abs(G),axis=1),1.)
    G=G/scale[:,None];h=h/scale
    G=np.vstack((G,np.zeros(n)));h=np.r_[h,-1.]
    M=np.column_stack((G,np.ones(len(h))))
    w=np.ones(n)/n if initial is None else (initial+1e-8)/(initial.sum()+n*1e-8)
    x=np.r_[w,max(float(np.max(h-G@w))+1,1.)]
    eq=np.r_[np.ones(n),0.];history=[]
    for stage in range(24):
        mu=.1*(.2**stage)
        for iteration in range(60):
            slack=M@x-h
            if np.min(slack)<=0 or np.min(x[:-1])<=0:raise ValueError('Lost interior')
            inv=1/slack
            grad=np.r_[np.zeros(n),1.]-mu*(M.T@inv+np.r_[1/x[:-1],0.])
            H=mu*(M.T@(M*inv[:,None]**2))
            H[np.arange(n),np.arange(n)]+=mu/x[:-1]**2
            try:
                sol=np.linalg.solve(H,np.column_stack((-grad,eq)))
            except np.linalg.LinAlgError:
                return x[:-1],history,'singular Newton system'
            step=sol[:,0]-sol[:,1]*(eq@sol[:,0])/(eq@sol[:,1])
            descent=float(grad@step)
            if -descent/2 < max(mu*1e-7,1e-24):break
            def objective(z):
                s=M@z-h
                if min(z[:-1])<=0 or min(s)<=0:return float('inf')
                return float(z[-1]-mu*(np.log(z[:-1]).sum()+np.log(s).sum()))
            old=objective(x);alpha=1.
            while alpha>1e-16 and objective(x+alpha*step)>old+.01*alpha*descent:alpha*=.5
            if alpha<=1e-16:break
            x+=alpha*step
        history.append({'stage':stage,'mu':mu,'phase_one_slack':float(x[-1]),'sum_weights':float(x[:-1].sum())})
    return x[:-1],history,'finished barrier schedule'
