import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp
from utilities.odes import dm_dt
from utilities.func_rk4 import rk4
from utilities.euler_method import euler_method


def plot_integrator_results(title_string, args: tuple):
    # Time interval for integration
    t_min = 1850
    t_max = 2300
    # Number of points in time array (only used for rk4)
    n = 10000
    # max_step = 10
    max_step = 1e-3  # dialed max_step down for nine boxes
    m_init, k, add_flux_c, add_emissions = args

    # todo: pickle these results for 4- and 9-boxes
    t1 = time.time()
    # t_rk4, M_rk4 = rk4(fxy=dm_dt, x0=t_min, xf=t_max, y0=m_init,
    #                    N=n, args=(k, add_flux_c, add_emissions))

    t2 = time.time()
    # rk23_sol = solve_ivp(fun=dm_dt, t_span=(t_min, t_max), y0=m_init,
    #                      method="RK23", max_step=max_step, args=(k, add_flux_c, add_emissions))

    t3 = time.time()
    # rk45_sol = solve_ivp(fun=dm_dt, t_span=(t_min, t_max), y0=m_init,
    #                      method="RK45", max_step=max_step, args=(k, add_flux_c, add_emissions))

    t4 = time.time()
    # dop853_sol = solve_ivp(fun=dm_dt, t_span=(t_min, t_max), y0=m_init,
    #                        method="DOP853", max_step=max_step, args=(k, add_flux_c, add_emissions))

    t5 = time.time()
    lsoda_sol = solve_ivp(fun=dm_dt, t_span=(t_min, t_max), y0=m_init,
                          method="LSODA", max_step=max_step, args=(k, add_flux_c, add_emissions))

    t6 = time.time()
    # bdf_sol = solve_ivp(fun=dm_dt, t_span=(t_min, t_max), y0=m_init,
    #                     method="BDF", max_step=max_step, args=(k, add_flux_c, add_emissions))
    # t_euler, M_euler = euler_method(fxy=dm_dt, x0=t_min, xf=t_max, y0=M_init, N=n)
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
    # all_t = [rk23_sol.t, rk45_sol.t, dop853_sol.t, lsoda_sol.t, bdf_sol.t]
    # all_M = [rk23_sol.y.T, rk45_sol.y.T, dop853_sol.y.T, lsoda_sol.y.T, bdf_sol.y.T]
    # deltat = [deltat_rk23, deltat_rk45, deltat_dop853, deltat_lsoda, deltat_bdf]
    # plot_types = ["RK23", "RK45", "DOP853", "LSODA", "BDF"]
    all_t = [lsoda_sol.t, ]
    all_M = [lsoda_sol.y.T]
    deltat = [deltat_lsoda]
    plot_types = ["LSODA"]
    # all_t = [t_rk4, rk23_sol.t, rk45_sol.t, dop853_sol.t, lsoda_sol.t, t_euler]
    # all_M = [M_rk4.real, rk23_sol.y.T, rk45_sol.y.T, dop853_sol.y.T, lsoda_sol.y.T, M_euler]
    # deltat = [deltat_rk4, deltat_rk23, deltat_rk45, deltat_dop853, deltat_lsoda, deltat_euler]
    # plot_types = ["(Given) RK4", "RK23", "RK45", "DOP853", "LSODA", "Euler"]

    # Plotting
    # fig, ax = plt.subplots()
    fig = plt.figure(figsize=(9, 4), dpi=150)

    for ii in range(len(plot_types)):
        # plt.subplot(3, 2, ii + 1)
        plt.plot(all_t[ii], all_M[ii][:, [0, 1]])  # plotting [0, 1] for just forced atmosphere and surface ocean water
        plt.xlabel('Time (yr)')
        plt.ylabel('Mass (Gt)')
        plt.title(plot_types[ii] + ", delta t = " + "{:.2E}".format(deltat[ii]) + "s")

    plt.suptitle(title_string)
    fig.legend(['atmosphere',
                'surface water',
                # 'surface biota',
                # 'intermediate and deep water',
                # 'short-lived biota',
                # 'long-lived biota',
                # 'litter',
                # 'soil', 'peat'
                ], loc="center right")
    plt.tight_layout()

    plt.show()
