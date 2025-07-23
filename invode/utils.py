# utils.py

import numpy as np
from scipy.optimize import minimize
import scipy.io


def load_matlab_data(file_path):
    data = scipy.io.loadmat(file_path)
    return data


def local_refine(best_params, ode_func, error_func, fixed_params, param_bounds, method='L-BFGS-B', verbose=False):
    """
    Perform local gradient-based optimization to refine parameter estimates.
    
    This function takes a set of promising parameter values (typically from global
    optimization) and applies local optimization techniques to find a nearby local
    optimum with higher precision. It uses gradient-based methods that can efficiently
    navigate smooth objective function landscapes to achieve fine-scale convergence.
    
    The function is designed to work seamlessly with mixed parameter sets containing
    both varying (free) and fixed parameters. Only the free parameters are optimized,
    while fixed parameters remain constant throughout the refinement process.
    
    Local refinement is particularly effective when:
    
    - The objective function is smooth and differentiable near the optimum
    - Global optimization has identified a promising region
    - High precision is required for the final parameter estimates
    - Gradient information is available or can be approximated numerically
    
    Parameters
    ----------
    best_params : dict
        Initial parameter estimates to be refined. Should contain values for all
        parameters required by the ODE function. These typically come from a
        global optimization phase and represent a good starting point for local
        search.
        
        Example: ``{'k1': 2.5, 'k2': 0.15, 'alpha': 0.8, 'beta': 1.2}``
        
    ode_func : callable
        The ODE solver function that takes a parameter dictionary and returns
        model output. Must have signature ``ode_func(params) -> output`` where
        params is a dictionary of parameter values and output is the model
        prediction that will be compared against data.
        
    error_func : callable  
        Error/objective function that quantifies model fit quality. Must have
        signature ``error_func(model_output) -> float`` where model_output is
        the result from ode_func and the return value is a scalar error measure
        (lower values indicate better fit).
        
    fixed_params : dict
        Dictionary of parameters that should remain constant during optimization.
        These parameters will be included in every evaluation but their values
        will not be modified by the local optimizer.
        
        Example: ``{'temp': 298.15, 'pH': 7.4}``
        
    param_bounds : dict
        Dictionary mapping parameter names to their allowable bounds. Each value
        should be a tuple (min_val, max_val). Only bounds for free parameters
        (those not in fixed_params) are used during optimization.
        
        Example: ``{'k1': (0.1, 10.0), 'k2': (0.01, 1.0)}``
        
    method : str, optional
        Optimization algorithm to use. Must be a method supported by 
        scipy.optimize.minimize that handles bound constraints. Common choices:
        
        - ``'L-BFGS-B'`` (default): Limited-memory BFGS with bounds
        - ``'TNC'``: Truncated Newton with bounds  
        - ``'SLSQP'``: Sequential Least Squares Programming
        - ``'trust-constr'``: Trust-region constrained optimization
        
        Default is 'L-BFGS-B' which provides good performance for smooth problems.
        
    verbose : bool, optional
        If True, print detailed information about the optimization process,
        including refined parameter values, final error, and success status.
        Useful for debugging and monitoring convergence. Default is False.
    
    Returns
    -------
    tuple of (dict, float)
        A tuple containing the refined results:
        
        - **refined_params** (dict): The optimized parameter set containing both
          refined free parameters and unchanged fixed parameters. Has the same
          keys as the input best_params.
        - **refined_error** (float): The error value achieved by the refined
          parameters. This should be less than or equal to the initial error
          if optimization was successful.
    
    Notes
    -----
    **Algorithm Details:**
    
    The function implements a parameter space transformation to work with scipy's
    optimize interface:
    
    1. Separates free parameters (to be optimized) from fixed parameters
    2. Creates a wrapper function that converts parameter vectors back to dictionaries
    3. Merges free and fixed parameters for each function evaluation
    4. Handles exceptions gracefully by returning infinite error for failed evaluations
    
    **Error Handling:**
    
    The wrapped objective function includes robust exception handling. If the ODE
    solver or error function raises any exception (numerical instability, domain
    errors, etc.), the function returns np.inf, allowing the optimizer to continue
    with other parameter values.
    
    **Convergence Criteria:**
    
    The optimization uses scipy's default convergence criteria for the selected
    method. These typically include:
    
    - Gradient norm tolerance
    - Function value change tolerance  
    - Maximum iteration limits
    - Constraint satisfaction tolerance
    
    **Parameter Bounds:**
    
    All free parameters are constrained to remain within their specified bounds
    throughout the optimization process. The optimizer will not evaluate parameter
    values outside these bounds.
    
    Examples
    --------
    Basic usage after global optimization:
    
    >>> # Assume we have results from global optimization
    >>> global_best = {'k1': 2.3, 'k2': 0.18, 'temp': 298.15}
    >>> fixed = {'temp': 298.15}  # Temperature held constant
    >>> bounds = {'k1': (0.1, 10.0), 'k2': (0.01, 1.0)}
    >>> 
    >>> refined_params, refined_error = local_refine(
    ...     global_best, my_ode_solver, my_error_function, 
    ...     fixed, bounds, verbose=True
    ... )
    >>> print(f"Refined parameters: {refined_params}")
    >>> print(f"Final error: {refined_error:.6f}")
    
    Comparing different optimization methods:
    
    >>> methods = ['L-BFGS-B', 'TNC', 'SLSQP']
    >>> results = {}
    >>> 
    >>> for method in methods:
    ...     params, error = local_refine(
    ...         initial_params, ode_func, error_func,
    ...         fixed_params, param_bounds, method=method
    ...     )
    ...     results[method] = (params, error)
    ...     print(f"{method}: error = {error:.6f}")
    
    Integration with global optimization workflow:
    
    >>> def optimize_ode_model(ode_func, error_func, param_config):
    ...     # Global optimization phase
    ...     global_optimizer = ODEOptimizer(
    ...         ode_func=ode_func,
    ...         error_func=error_func,
    ...         **param_config
    ...     )
    ...     global_best, global_error = global_optimizer.fit()
    ...     
    ...     # Local refinement phase
    ...     refined_best, refined_error = local_refine(
    ...         global_best, ode_func, error_func,
    ...         param_config['fixed_params'],
    ...         param_config['param_bounds'],
    ...         method='L-BFGS-B',
    ...         verbose=True
    ...     )
    ...     
    ...     improvement = global_error - refined_error
    ...     print(f"Local refinement improved error by {improvement:.6f}")
    ...     return refined_best, refined_error
    
    Handling optimization failures:
    
    >>> try:
    ...     refined_params, refined_error = local_refine(
    ...         best_params, ode_func, error_func, fixed_params, bounds
    ...     )
    ...     if refined_error == np.inf:
    ...         print("Warning: Local optimization failed to find valid solution")
    ...         # Fall back to global optimization result
    ...         refined_params = best_params
    ... except Exception as e:
    ...     print(f"Local refinement failed: {e}")
    ...     refined_params = best_params  # Use original parameters
    
    Performance Tips
    ----------------
    **Method Selection:**
    
    - Use ``'L-BFGS-B'`` for smooth, well-behaved problems (default choice)
    - Use ``'TNC'`` for problems with more complex constraint handling needs
    - Use ``'trust-constr'`` for highly nonlinear problems or when high accuracy is needed
    - Avoid gradient-free methods unless the objective function is non-smooth
    
    **Numerical Stability:**
    
    - Ensure ODE solver uses appropriate tolerances for the refinement scale
    - Consider parameter scaling if parameters have very different magnitudes
    - Monitor for numerical issues in verbose mode
    
    **Computational Efficiency:**
    
    - Local refinement is typically fast (10-100 function evaluations)
    - Most time is spent in ODE solving, not optimization overhead
    - Consider parallel local refinement of multiple candidates
    
    Troubleshooting
    ---------------
    **Common Issues:**
    
    - **Optimization not improving**: Check that the starting point is reasonable
      and within bounds. Verify the objective function is smooth locally.
    - **Numerical errors**: Reduce ODE solver tolerances or adjust parameter bounds
      to avoid regions where the model becomes unstable.
    - **Slow convergence**: Try different optimization methods or check for
      ill-conditioning in the parameter space.
    
    See Also
    --------
    scipy.optimize.minimize : The underlying optimization function
    ODEOptimizer.fit : Global optimization method that often precedes local refinement
    lhs_sample : Parameter sampling function for global search phase
    
    References
    ----------
    .. [1] Nocedal, J., & Wright, S. (2006). "Numerical optimization" 
           (Vol. 2). New York: Springer.
    .. [2] Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). "A limited 
           memory algorithm for bound constrained optimization." SIAM Journal 
           on scientific computing, 16(5), 1190-1208.
    .. [3] Liu, D. C., & Nocedal, J. (1989). "On the limited memory BFGS 
           method for large scale optimization." Mathematical programming, 
           45(1-3), 503-528.
    """
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
