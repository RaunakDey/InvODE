# utils.py

import numpy as np

def shrink_bounds(center, bounds, rate):
    return {
        k: (
            max(bounds[k][0], center[k] - rate * (bounds[k][1] - bounds[k][0]) / 2),
            min(bounds[k][1], center[k] + rate * (bounds[k][1] - bounds[k][0]) / 2)
        )
        for k in bounds
    }

def check_bounds(initial, bounds):
    for k in initial:
        if not (bounds[k][0] <= initial[k] <= bounds[k][1]):
            raise ValueError(f"Initial guess for {k} is out of bounds")

def param_dict_to_array(param_dict, bounds):
    return np.array([param_dict[k] for k in bounds])

def array_to_param_dict(arr, bounds):
    return dict(zip(bounds.keys(), arr))
