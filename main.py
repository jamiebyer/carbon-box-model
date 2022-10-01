import argparse
import numpy as np
from utilities.plotters import plot_integrator_results

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

n_boxes = 4

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

# todo: migrate to config
# Plot integration for 4 box model

# add_flux_C = True
add_flux_C = True
# add_emissions = True
add_emissions = True

if add_flux_C:
    title_string += ", add sinusoidal carbon forcing"
if add_emissions:
    title_string += ", add emissions"


def main():
    plot_integrator_results(title_string, args=(M_init, k, add_flux_C, add_emissions))


if __name__ == '__main__':
    # todo: use argparse or config
    # argparse.
    main()
