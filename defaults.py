""" Module contains the dictionary for model default parameters"""

import numpy as np

DEFAULTS = {
    "object_mass": 0.1,
    "kp_prox": 2,
    "kp_middle": 2,
    "kp_dist": 1,
    "gravity": np.array([0, 0, -9.81]),
    "impratio": 10,
}
