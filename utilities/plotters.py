import matplotlib.pyplot as plt
import time
import json
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from utilities.odes import dm_dt
from utilities.func_rk4 import rk4
from utilities.euler_method import euler_method
from utilities.emissions import emissions


def plot_integrator_results(title_string, args: tuple):
    # Time interval for integration
    t_min = 1850
    t_max = 2300
    # Number of points in time array (used for rk4, euler)
    n = 10000
    # max_step = 10
    max_step = 1e-3  # dialed max_step down for nine boxes
    m_init, k, emissions_models, integrators = args

    # todo: pickle these results for 4- and 9-boxes
    all_t = []
    all_M = []
    delta_t = []
    plot_titles = []

    for integ in integrators:
        for model in emissions_models:
            if integ == "rk4":
                t_start = time.time()
                t, M = rk4(fxy=dm_dt, x0=t_min, xf=t_max, y0=m_init, N=n, args=(k, model))
                t_end = time.time()
            elif integ == "euler":
                t_start = time.time()
                t, M = euler_method(fxy=dm_dt, x0=t_min, xf=t_max, y0=m_init, N=n, args=(k, model))
                t_end = time.time()
            else:
                t_start = time.time()
                sol = solve_ivp(fun=dm_dt, t_span=(t_min, t_max), y0=m_init, method=integ, max_step=max_step, args=(k, model))
                t_end = time.time()
                t = sol.t
                M = sol.y.T
            
            all_t.append(t)
            all_M.append(M)
            delta_t.append(t_end-t_start)
            plot_titles.append(integ + ", " + model.replace("_", " "))

    # Plotting
    n_plots = len(plot_titles)
    if n_plots > 1:
        fig, ax = plt.subplots()
        if n_plots <= 3:
            n_rows = 1
            n_cols = n_plots
        else:
            n_cols = 3
            n_rows = int(np.ceil(n_plots/n_cols))

        for ii in range(n_plots):
            plt.subplot(n_rows, n_cols, ii + 1)
            plt.plot(all_t[ii], all_M[ii][:, [0, 1]])  # plotting [0, 1] for just forced atmosphere and surface ocean water
            plt.xlabel('Time (yr)')
            plt.ylabel('Mass (Gt)')
            plt.title(plot_titles[ii] + ", delta t = " + "{:.2E}".format(delta_t[ii]) + "s")
    else:
        fig = plt.figure(figsize=(9, 4), dpi=150)
        plt.plot(all_t[0], all_M[0][:, [0, 1]])  # plotting [0, 1] for just forced atmosphere and surface ocean water
        plt.xlabel('Time (yr)')
        plt.ylabel('Mass (Gt)')
        plt.title(plot_titles[0] + ", delta t = " + "{:.2E}".format(delta_t[0]) + "s")

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
    plt.savefig(f"figures/nine_box_long_wavelength_v1.png")

    plt.show()

def plot_emissions_models():
    with open('data/models/models.json') as json_file:
        models = json.load(json_file)

    df = pd.read_csv("./data/model-data.csv")
    
    plt.subplot(1, 2, 1)
    t = np.linspace(2010, 2100, 1000)
    #t = np.linspace(0, 100, 1000)
    eq_names = ["short_sine", "long_sine", "short_exp", "long_exp"]
    for model in eq_names:
        plt.plot(t, emissions(t, model))
    #plt.legend(["short sinusoidal", "long sinusoidal", "short exponential", "long exponential"])
    plt.title("Emissions from equations")

    plt.subplot(1, 2, 2)
    model_names = list(models.keys()) + ["IPCC-A2"]
    for model in model_names:
        mod_id, exp_id = model.split('_')
        plt.plot(df[model + "_times"], df[model + "_emissions"], alpha=0.5, label=mod_id+", "+exp_id)
    
    plt.legend()
    plt.xlim([2010, 2100])
    plt.title("Emissions from models")
    plt.xlabel("Time")
    plt.ylabel("Carbon emissions (Gt/yr)")
    
    plt.suptitle("Carbon emissions")
    plt.show()
    