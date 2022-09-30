import numpy as np

def euler_method(fxy, x0, xf, y0, N):
    dt = (xf-x0)/N    
    t = np.linspace(x0, xf, N)
    y = np.empty((N, len(y0)))
    y[0] = y0
    
    #march forward in time
    for ii in range(len(t)-1):
        y[ii+1] = y[ii] + fxy(t[ii], y[ii])*dt
        
    return t, y