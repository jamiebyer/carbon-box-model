from utilities.emissions import emissions

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import time

# STEP 0.1 -
# Initial values in boxes in Gt
M1 = 725
M2 = 725
M5 = 110
M7 = 60
M3 = 3
M4 = 376.75
M6 = 450
M8 = 1350
M9 = 160
# Array of initial masses for each box.


# # Fluxes in Gt/yr (Fab is the flux from box a to box b)
# F12 = 90
# F21 = 90
# F71 = 55
# F57 = 55
# F15 = 110
# F51 = 55
# F72 = 0
#
# F11, F22, F52, F25, F55, F75, F17, F27, F77 = np.zeros(9)
#
# # F matrix
# F_init = [
#     [F11, F21, F51, F71],
#     [F12, F22, F52, F72],
#     [F15, F25, F55, F75],
#     [F17, F27, F57, F77],
# ]

n_boxes = 9

if n_boxes == 4:
    title_string = "four box model"
    print("computing four box model")
    M_init = np.array([M1, M2, M5, M7])
    F_init = np.genfromtxt('data/four_box_fluxes.csv',
                           delimiter=','
                           )
else:
    title_string = "nine box model"
    print("computing nine box model")
    M_init = np.array([M1, M2, M3, M4, M5, M6, M7, M8, M9])
    F_init = np.genfromtxt('data/nine_box_fluxes.csv',
                           delimiter=','
                           )

# Get rate coefficients from steady-state flux and initial mass
k = np.divide(F_init, M_init)


# todo: migrate to odes.py
# Function used by integrators
def dMdt(t, M):
    """
    t: float (scalar), time.
    M: size 4 or 9 array, masses of carbon in boxes 1, 2, 5, 7, (3, 4, 6, 8, 9).

    dMdt: size 4 or 9 array, change in mass for boxes 1, 2, 5, 7, (3, 4, 6, 8, 9).
    """

    # Update values for fluxes from mass.
    F = np.multiply(k, M)

    F_in = np.sum(F, axis=1)  # row-wise for in-flux
    F_out = np.sum(F, axis=0)  # column-wise for out-flux

    dMdt = F_in - F_out

    # From step 0.1, add an arbitrary flux C (in Gt/yr)
    if add_flux_C:
        T = 0.1 / (2 * np.pi)
        # T = 100 / (2 * np.pi)
        C = (20 / T) * (np.sin(t / T))
    else:
        C = 0

    if add_emissions:
        e = emissions([t])[0]
    else:
        e = 0

    # add forcing fluxes to first slot: the atmosphere box
    dMdt[0] += C + e

    return dMdt


# todo: migrate to plotters.py
def plot_integrator_results(title_string):
    # Time interval for integration
    t_min = 0
    t_max = 2500
    # Number of points in time array (only used for rk4)
    n = 1000
    max_step = 1e-2

    t1 = time.time()
    # t_rk4, M_rk4 = rk4(fxy=dMdt, x0=t_min, xf=t_max, y0=M_init, N=n)

    t2 = time.time()
    rk23_sol = solve_ivp(fun=dMdt, t_span=(t_min, t_max), y0=M_init, method="RK23", max_step=max_step)

    t3 = time.time()
    rk45_sol = solve_ivp(fun=dMdt, t_span=(t_min, t_max), y0=M_init, method="RK45", max_step=max_step)

    t4 = time.time()
    dop853_sol = solve_ivp(fun=dMdt, t_span=(t_min, t_max), y0=M_init, method="DOP853", max_step=max_step)

    t5 = time.time()
    lsoda_sol = solve_ivp(fun=dMdt, t_span=(t_min, t_max), y0=M_init, method="LSODA", max_step=max_step)

    t6 = time.time()
    bdf_sol = solve_ivp(fun=dMdt, t_span=(t_min, t_max), y0=M_init, method="BDF", max_step=max_step)
    # t_euler, M_euler = euler_method(fxy=dMdt, x0=t_min, xf=t_max, y0=M_init, N=n)
    t7 = time.time()

    # Get time taken for each integrator to perform integration.
    deltat_rk4 = t2 - t1
    deltat_rk23 = t3 - t2
    deltat_rk45 = t4 - t3
    deltat_dop853 = t5 - t4
    deltat_lsoda = t6 - t5
    deltat_bdf = t7 - t6
    deltat_euler = t7 - t6

    # Arrays to loop over and plot each integration type.
    all_t = [rk23_sol.t, rk45_sol.t, dop853_sol.t, lsoda_sol.t, bdf_sol.t]
    all_M = [rk23_sol.y.T, rk45_sol.y.T, dop853_sol.y.T, lsoda_sol.y.T, bdf_sol.y.T]
    deltat = [deltat_rk23, deltat_rk45, deltat_dop853, deltat_lsoda, deltat_bdf]
    plot_types = ["RK23", "RK45", "DOP853", "LSODA", "BDF"]
    # all_t = [t_rk4, rk23_sol.t, rk45_sol.t, dop853_sol.t, lsoda_sol.t, t_euler]
    # all_M = [M_rk4.real, rk23_sol.y.T, rk45_sol.y.T, dop853_sol.y.T, lsoda_sol.y.T, M_euler]
    # deltat = [deltat_rk4, deltat_rk23, deltat_rk45, deltat_dop853, deltat_lsoda, deltat_euler]
    # plot_types = ["(Given) RK4", "RK23", "RK45", "DOP853", "LSODA", "Euler"]

    # Plotting    
    fig, ax = plt.subplots()

    for ii in range(len(plot_types)):
        plt.subplot(3, 2, ii + 1)
        plt.plot(all_t[ii], all_M[ii])
        plt.xlabel('Time (yr)')
        plt.ylabel('Mass (Gt)')
        plt.title(plot_types[ii] + ", delta t = " + "{:.2E}".format(deltat[ii]) + "s")

    plt.suptitle(title_string)
    fig.legend(['atmosphere',
                'surface water',
                'surface biota',
                'intermediate and deep water',
                'short-lived biota',
                'long-lived biota',
                'litter',
                'soil', 'peat'], loc="lower right")
    plt.tight_layout()

    plt.show()


# todo: migrate to config
# Plot integration for 4 box model

# add_flux_C = True
add_flux_C = True
# add_emissions = True
add_emissions = True

if add_flux_C:
    title_string += ", add sinusoidal flux C"
if add_emissions:
    title_string += ", add emissions"


def main():
    plot_integrator_results(title_string)


if __name__ == '__main__':
    main()
