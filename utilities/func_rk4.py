#    RK4.M   Use 4th order Runge Kutta Method for Numerical Solution of IVPs

# The inputs to the function are:
#         fxy = the name of the function containing f(x,y) (e.g. oneode, twoode)
#         xo,xf = initial and final values of the independent variable (integers or floats)
#         yo = initial value of dependent variable at xo (numpy array)
#         N = number of intervals to use between xo and xf (integer)

# The outputs to the function are:
#         X = numpy array containing values of the independent variable
#         Y = the estimated dependent variable at each value of the independent variable
#         --> this variable is a 1D numpy array if only one equation is solved
#         --> it is an M-D numpy array [y1(x) y2(x) ... ] for multiple (M) equations 

import numpy as np


def rk4(fxy, x0, xf, y0, N, args):

    # compute step size and size of output variables
    if N < 2:
        N = 2  # set minimum number for N
    h = (xf - x0) / N
    X = np.zeros((N+1, 1))
    M = np.max(np.shape(y0))
    Y = np.zeros((N+1, M))*1j  # make complex by multiplying by 1j;
    # this way can add complex values to this during integration

    # set initial conditions
    x = x0
    X[0] = x
    y = [complex(val) for val in y0]  # make complex
    Y[0,:] = y
    
    # begin computational loop
    for ii in range(N):
        # evaluate function fxy; depending on equation, k1-4 can be complex; this is why we make Y and y complex as well
        k1 = np.array([h * val for val in fxy(x, y, *args)])  # todo: include additional params in fxy call
        k2 = np.array([h * val for val in fxy(x+h/2, y+k1/2, *args)])
        k3 = np.array([h * val for val in fxy(x+h/2, y+k2/2, *args)])
        k4 = np.array([h * val for val in fxy(x+h, y+k3, *args)])
        
        y += (k1 + 2*k2 + 2*k3 + k4) / 6.
        x += h
        X[ii+1] = x
        Y[ii+1, :] = y
    
    return X, Y
