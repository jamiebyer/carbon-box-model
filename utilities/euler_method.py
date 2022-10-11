import numpy as np

def euler_method(fxy, x0, xf, y0, N, args):
    """
    Function to solve ODEs using Euler's method.
    """
    # get time step from initial and final time, and number of steps
    dt = (xf-x0)/N    
    t = np.linspace(x0, xf, N)
    y = np.empty((N, len(y0)))
    y[0] = y0
    
    # loop over each time in t and calculate y at each step
    for ii in range(len(t)-1):
        y[ii+1] = y[ii] + fxy(t[ii], y[ii], *args)*dt
        
    return t, y
