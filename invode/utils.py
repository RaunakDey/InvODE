# utils.py



import numpy as np
from scipy.optimize import minimize
import scipy.io

def local_refine(best_params, ode_func, error_func, param_bounds, method='L-BFGS-B', verbose=False):
    param_keys = list(best_params.keys())

    def wrapped_error(param_vector):
        param_dict = dict(zip(param_keys, param_vector))
        try:
            output = ode_func(param_dict)
            return error_func(output)
        except:
            return np.inf

    initial_vector = [best_params[k] for k in param_keys]
    bounds_list = [param_bounds[k] for k in param_keys]

    result = minimize(wrapped_error, x0=initial_vector, method=method, bounds=bounds_list)

    if verbose:
        print("\n[Local Optimization]")
        if result.success:
            print("Refined parameters:", dict(zip(param_keys, result.x)))
            print("Refined error:", result.fun)
        else:
            print("Local optimization failed:", result.message)

    return dict(zip(param_keys, result.x)), result.fun








def load_matlab_data(file_path):
    data = scipy.io.loadmat(file_path)
    return data





'''
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

'''
