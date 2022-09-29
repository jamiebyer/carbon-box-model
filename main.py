from emissions import emissions
from func_rk4 import rk4

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import time


# STEP 0.1
# Initial values in boxes in Gt
M1_init = 725
M2_init = 725
M5_init = 110
M7_init = 60
# Array of initial masses for each box.
M_init = np.array([M1_init, M2_init, M5_init, M7_init])

# Fluxes in Gt/yr (Fab is the flux from box a to box b)
F12_init = 90
F21_init = 90
F71_init = 55
F57_init = 55
F15_init = 110
F51_init = 55
F72_init = 0

# Get rate coefficients from steady-state flux and initial mass
k12 = F12_init/M1_init
k21 = F21_init/M2_init
k71 = F71_init/M7_init
k57 = F57_init/M5_init
k15 = F15_init/M1_init
k51 = F51_init/M5_init
k72 = F72_init/M2_init


# Function used integrators
def dMdt(t, M):
    """
    t: float, time.
    M: size 4 array, mass of carbon in boxes 1, 2, 5, 7.

    dMdt: size 4 array, change in mass for boxes 1, 2, 5, 7.
    """
    M1, M2, M5, M7 = M

    # Update values for fluxes from mass.
    F12 = k12*M1
    F21 = k21*M2
    F71 = k71*M7
    F57 = k57*M5
    F15 = k15*M1
    F51 = k51*M5
    F72 = k72*M7

    # From step 0.1, add an arbitraty flux C (in Gt/yr)
    if add_flux_C:
        C = 15
    else:
        C = 0

    if add_emissions:
        e = emissions([t])[0]
    else:
        e = 0

    # dMdt = flux in - flux out
    dM1dt = (F21 + F51 + F71 + C + e) - (F12 + F15) # Add emissions to the atmosphere as part of flux in
    dM2dt = (F12 + F72) - (F21)
    dM5dt = (F15) - (F51 + F57)
    dM7dt = (F57) - (F71 + F72)

    dMdt = np.array([dM1dt, dM2dt, dM5dt, dM7dt])
    
    return dMdt

def plot_integrator_results(title_string):
    # Time interval for integration
    t_min = 0
    t_max = 2500
    # Number of points in time array (only used for rk4)
    n = 1000
    max_step = 10

    t1 = time.time()
    t_rk4, M_rk4 = rk4(fxy=dMdt, x0=t_min, xf=t_max, y0=M_init, N=n)
    
    t2 = time.time()
    rk23_sol = solve_ivp(fun=dMdt, t_span=(t_min, t_max), y0=M_init, method="RK23", max_step=max_step)
    
    t3 = time.time()
    rk45_sol = solve_ivp(fun=dMdt, t_span=(t_min, t_max), y0=M_init, method="RK45", max_step=max_step)
    
    t4 = time.time()
    dop853_sol = solve_ivp(fun=dMdt, t_span=(t_min, t_max), y0=M_init, method="DOP853", max_step=max_step)
    
    t5 = time.time()
    lsoda_sol = solve_ivp(fun=dMdt, t_span=(t_min, t_max), y0=M_init, method="LSODA", max_step=max_step)
    t6 = time.time()

    # Get time taken for each integrator to perform integration.
    deltat_rk4 = t2-t1
    deltat_rk23 = t3-t2
    deltat_rk45 = t4-t3
    deltat_dop853 = t5-t4
    deltat_lsoda = t6-t5

    # Arrays to loop over and plot each integration type.
    all_t = [t_rk4, rk23_sol.t, rk45_sol.t, dop853_sol.t, lsoda_sol.t]
    all_M = [M_rk4.real, rk23_sol.y.T, rk45_sol.y.T, dop853_sol.y.T, lsoda_sol.y.T]
    deltat = [deltat_rk4, deltat_rk23, deltat_rk45, deltat_dop853, deltat_lsoda]
    plot_types = ["(Given) RK4", "RK23", "RK45", "DOP853", "LSODA"]

    # Plotting    
    fig, ax = plt.subplots()

    for ii in range(len(plot_types)):
        plt.subplot(3, 2, ii+1)
        plt.plot(all_t[ii], all_M[ii])
        plt.xlabel('Time (yr)')
        plt.ylabel('Mass (Gt)')
        plt.title(plot_types[ii] + ", delta t = " + "{:.2E}".format(deltat[ii]) + "s")

        #plt.axvline(x=1990)
        #plt.axvline(x=2100)
        #plt.xlim([1800, 2200])
    
    plt.suptitle(title_string)
    fig.legend(['atmosphere', 'surface water', 'short-lived biota', 'litter'], loc="lower right")
    plt.tight_layout()

    plt.show()


# Plot integration for 4 box model

#add_flux_C = True
add_flux_C = False
add_emissions = True
#add_emissions = False

title_string = "4 box model"

if add_flux_C:
    title_string += ", add flux C = 15 Gt/yr"
if add_emissions:
    title_string += ", add emissions"

plot_integrator_results(title_string)


