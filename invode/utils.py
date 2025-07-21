# utils.py

import numpy as np
from scipy.optimize import minimize
import scipy.io


def load_matlab_data(file_path):
    data = scipy.io.loadmat(file_path)
    return data


def local_refine(best_params, ode_func, error_func, fixed_params, param_bounds, method='L-BFGS-B', verbose=False):
    free_params = {k: v for k, v in best_params.items() if k not in fixed_params}
    free_param_keys = list(free_params.keys())

    def wrapped_error(param_vector):
        param_dict = dict(zip(free_param_keys, param_vector))
        full_param_dict = {**param_dict, **fixed_params}  # merge at every call!
        try:
            output = ode_func(full_param_dict)
            return error_func(output)
        except Exception as e:
            if verbose:
                print(f"Exception in wrapped_error: {e}")
            return np.inf

    initial_vector = [free_params[k] for k in free_param_keys]
    bounds_list = [param_bounds[k] for k in free_param_keys]

    result = minimize(wrapped_error, x0=initial_vector, method=method, bounds=bounds_list)

    if verbose:
        print("\n[Local Optimization]")
        if result.success:
            refined_params = dict(zip(free_param_keys, result.x))
            print("Refined parameters:", {**refined_params, **fixed_params})
            print("Refined error:", result.fun)
        else:
            print("Local optimization failed:", result.message)

    final_params = {**dict(zip(free_param_keys, result.x)), **fixed_params}
    return final_params, result.fun

'''

def local_refine3(best_params, ode_func, error_func, fixed_params, param_bounds, method='L-BFGS-B', verbose=False):
    # Identify which parameters are fixed (same low == high)
    #fixed_params = {k: v[0] for k, v in param_bounds.items() if isinstance(v, tuple) and v[0] == v[1]}

    free_params = {k: v for k, v in best_params.items() if k not in fixed_params}
    full_param_dict = {**free_params, **fixed_params}  # combine free and fixed

    free_param_keys = list(free_params.keys())
    
    def wrapped_error(param_vector):
        param_dict = dict(zip(free_param_keys, param_vector))
        #full_param_dict = {**param_dict, **fixed_params}  # combine free and fixed
        print("\nFull param:", full_param_dict)
        try:
            output = ode_func(full_param_dict)
            return error_func(output)
        except:
            return np.inf

    initial_vector = [free_params[k] for k in free_param_keys]
    bounds_list = [param_bounds[k] for k in free_param_keys]

    result = minimize(wrapped_error, x0=initial_vector, method=method, bounds=bounds_list)

    if verbose:
        print("\n[Local Optimization]")
        if result.success:
            refined_params = dict(zip(free_param_keys, result.x))
            print("Refined parameters:", {**refined_params, **fixed_params})
            print("Refined error:", result.fun)
        else:
            print("Local optimization failed:", result.message)

    # Return merged result
    final_params = {**dict(zip(free_param_keys, result.x)), **fixed_params}
    return final_params, result.fun


def local_refine2(best_params, ode_func, error_func, param_bounds, method='L-BFGS-B', verbose=True):

    param_keys = list(best_params.keys())
    print("\n[Local Optimization] Initial parameters:", best_params)
    print("Parameter bounds:", param_bounds)

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
