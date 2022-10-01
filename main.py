import argparse
import os

import numpy as np
from utilities.plotters import plot_integrator_results

n_boxes = 4

if n_boxes == 4:
    title_string = "four box model"
    print("computing four box model")
    initial_masses = np.genfromtxt('data/four_initial_masses.csv',
                                   delimiter=',')
    initial_fluxes = np.genfromtxt('data/four_box_fluxes.csv',
                                   delimiter=','
                                   )
else:
    title_string = "nine box model"
    print("computing nine box model")
    initial_masses = np.genfromtxt('data/initial_masses.csv',
                                   delimiter=',')
    initial_fluxes = np.genfromtxt('data/nine_box_fluxes.csv',
                                   delimiter=','
                                   )

# Get rate coefficients from steady-state flux and initial mass
k = np.divide(initial_fluxes, initial_masses)

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
    plot_integrator_results(title_string, args=(initial_masses, k, add_flux_C, add_emissions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("n_boxes",
                        help='choose how many boxes for carbon box model; default is 4',
                        type=int)
    parser.add_argument("initial_masses", type=os.PathLike)
    parser.add_argument("initial_fluxes", type=os.PathLike)
    # todo: option to add either sinusoidal or exponential damp forcing
    parser.add_argument("--add_flux_C", action='store_true', type=bool)
    parser.add_argument("--add_emissions", action='store_true', type=bool)
    parser.parse_args()
    main()
