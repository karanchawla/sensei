from copy import deepcopy
import numpy as np


def convert_to_si_units(params):
    params_in_si_units = deepcopy(params)

    # deg/root-hour -> (rad/s)/root-Hz
    params_in_si_units["angle_random_walk"] = 1.0 / \
        60.0 * np.deg2rad(params["angle_random_walk"])

    # deg/s/root-Hz —> rad/s/root-Hz
    params_in_si_units["noise_density"] = np.deg2rad(
        params["noise_density"])

    # deg/s -> rad/s
    params_in_si_units["dynamic_bias"] = np.deg2rad(params["dynamic_bias"])
    params_in_si_units["static_bias"] = np.deg2rad(params["static_bias"])

    return params_in_si_units
