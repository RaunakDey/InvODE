import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor

def lhs_sample(param_bounds, n_samples, seed=None):
    """
    Generate Latin Hypercube Samples within specified bounds for each parameter.

    Parameters:
    ----------
    param_bounds : dict
        Dictionary where keys are parameter names and values are (min, max) tuples.
    n_samples : int
        Number of samples to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns:
    -------
    samples : list of dict
        List of dictionaries containing sampled parameter sets.
    """
    keys = list(param_bounds.keys())
    bounds = np.array([param_bounds[k] for k in keys])  # shape: (n_params, 2)
    
    # Create a Latin Hypercube Sampler
    sampler = qmc.LatinHypercube(d=len(keys), seed=seed)
    unit_samples = sampler.random(n=n_samples)  # values in [0, 1]
    
    # Scale samples to actual bounds
    scaled_samples = qmc.scale(unit_samples, bounds[:, 0], bounds[:, 1])
    
    # Convert to list of dicts for readability
    samples = [dict(zip(keys, sample)) for sample in scaled_samples]
    
    return samples


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




def optimization(
    ode_func,
    error_func,
    param_bounds,
    initial_guess,
    n_samples=100,
    num_iter=5,
    shrink_factor=0.5,
    seed=None,
    verbose=False,
    verbose_plot=False,
    do_local_opt=False,
    local_method='L-BFGS-B',
    num_top_candidates=1,
    show_final_candidates=False,
    parallel=False,
    max_workers=None,  # uses all available by default
    local_parallel=False
    ):
    """
    Naively optimize parameters by iterated Latin Hypercube Sampling with shrinking bounds.

    Parameters:
    ----------
    ode_func : callable
        ODE simulation function. Should accept parameters as dict.
    error_func : callable
        Error function comparing model output to data. Should accept model output and return float.
    param_bounds : dict
        Initial parameter bounds. {param_name: (min, max)}
    initial_guess : dict
        Initial guess for parameters. Used to center bounds during shrinking.
    n_samples : int
        Number of samples per iteration.
    num_iter : int
        Number of iterations.
    shrink_factor : float
        Fraction by which bounds shrink in each iteration.
    seed : int or None
        Seed for reproducibility.
    verbose : bool
        Whether to print progress.

    Returns:
    -------
    best_params : dict
        Parameters with the lowest error found.
    best_error : float
        Corresponding error.
    """

    rng = np.random.default_rng(seed)

    # Validate initial guess is within bounds
    for key, value in initial_guess.items():
        if key not in param_bounds:
            raise ValueError(f"Initial guess includes unknown parameter '{key}'.")
        low, high = param_bounds[key]
        if not (low <= value <= high):
            raise ValueError(
                f"Initial guess for '{key}' = {value} is outside the bounds ({low}, {high})."
            )
        
    best_params = initial_guess.copy()
    best_error = float('inf')
    error_history = []
    top_candidates = [(initial_guess.copy(), float('inf'))]

    for iteration in range(num_iter):
        if verbose:
            print(f"\nIteration {iteration+1}/{num_iter}")

        all_sampled = []

        for candidate_params, _ in top_candidates:
            local_bounds = {}
            for key in param_bounds:
                full_min, full_max = param_bounds[key]
                width = (full_max - full_min) * (shrink_factor / 2)
                center = candidate_params[key]
                new_min = max(center - width, full_min)
                new_max = min(center + width, full_max)
                local_bounds[key] = (new_min, new_max)

            local_samples = lhs_sample(local_bounds, n_samples, seed=rng.integers(1e9))
            all_sampled.extend(local_samples)

        '''
        evaluated = []
        for param_set in all_sampled:
            try:
                output = ode_func(param_set)
                err = error_func(output)
                evaluated.append((param_set, err))
            except Exception as e:
                if verbose:
                    print(f"Sample failed: {e}")
                continue
        '''
        evaluated = []

        def evaluate(param_set):
            try:
                output = ode_func(param_set)
                err = error_func(output)
                return (param_set, err)
            except Exception:
                return None  # Skip failed ones

        if parallel:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = executor.map(evaluate, all_sampled)
                evaluated = [res for res in results if res is not None]
        else:
            evaluated = [res for res in map(evaluate, all_sampled) if res is not None]
            # evaluated = [evaluate(param_set) for param_set in all_sampled if evaluate(param_set) is not None]        

        evaluated.sort(key=lambda x: x[1])
        top_candidates = evaluated[:num_top_candidates]

        if top_candidates[0][1] < best_error:
            best_params = top_candidates[0][0]
            best_error = top_candidates[0][1]

        error_history.append(best_error)

        if verbose:
            print(f"Best error so far: {best_error}")
            print(f"Best params: {best_params}")

    '''
    ## post local optimization
    if do_local_opt:
        best_params, best_error = local_refine(best_params, ode_func, error_func, param_bounds, method=local_method, verbose=verbose)
    '''
    


    if do_local_opt:
        if local_parallel:
            from concurrent.futures import ProcessPoolExecutor
            def local_worker(p):
                return local_refine(p[0], ode_func, error_func, param_bounds, method=local_method, verbose=False)

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                refined_candidates = list(executor.map(local_worker, top_candidates))

            # Re-attach param values (since local_refine returns (param, error))
            refined_candidates = list(refined_candidates)
        else:
            refined_candidates = []
            for i, (params, err) in enumerate(top_candidates):
                refined_param, refined_error = local_refine(
                    params, ode_func, error_func, param_bounds,
                    method=local_method, verbose=verbose
                )
                refined_candidates.append((refined_param, refined_error))
                
        '''
        refined_candidates = []
        for i, (params, err) in enumerate(top_candidates):
            if verbose:
                print(f"\n[Local Optimization] Refining candidate {i+1}/{len(top_candidates)}")
            refined_param, refined_error = local_refine(
                params, ode_func, error_func, param_bounds,
                method=local_method, verbose=verbose
            )
            refined_candidates.append((refined_param, refined_error))
        '''
        

    # Sort again and update best
    refined_candidates.sort(key=lambda x: x[1])
    best_params, best_error = refined_candidates[0]

    ## show final candidates
    if verbose and show_final_candidates:
        print("\nTop candidates at final iteration:")
        for i, (params, err) in enumerate(top_candidates):
            print(f"  {i+1}. Error: {err:.6f}, Params: {params}")
            
    ## show plots
    if verbose_plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(error_history, marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Best Error")
        plt.title("Error over Optimization Iterations")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return best_params, best_error



