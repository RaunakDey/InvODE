# ode_fit/optimizer.py
from scipy.optimize import minimize
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from .sampling import lhs_sample
from .utils import local_refine
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import trange  


class ODEOptimizer:
    """
    A class for optimizing parameters of an ordinary differential equation (ODE) model.
    This class implements a sophisticated parameter optimization strategy that combines
    global exploration with local refinement. It uses Latin Hypercube Sampling (LHS)
    around the best candidates from each iteration, progressively shrinking search regions
    to converge on optimal parameter values. The algorithm maintains diversity by tracking
    multiple top candidates and optionally performs local optimization for fine-tuning.
    """


    def __init__(
        self,
        ode_func,
        error_func,
        param_bounds,
        initial_guess=None,
        n_samples=100,
        num_iter=10,
        num_top_candidates=1,
        do_local_opt=True,
        local_method='L-BFGS-B',
        shrink_rate=0.5,
        parallel=False,
        local_parallel=False,
        verbose=False,
        verbose_plot=False,
        seed=None,
        fixed_params=None
    ):
        
        """
        Initializes the optimizer for solving ordinary differential equations (ODEs).
        Parameters:
        -----------
        ode_func : callable
            The function representing the ODE to be solved.
        error_func : callable
            The function used to calculate the error between the model and the observed data.
        param_bounds : dict
            A dictionary defining the bounds for the parameters to be optimized. 
            Each key should correspond to a parameter name, and the value should be a tuple 
            (lower_bound, upper_bound) or a scalar for fixed parameters.
        initial_guess : dict, optional
            A dictionary providing the initial guess for the parameters. 
            Keys should match those in `param_bounds`. If not provided, a midpoint guess 
            will be used for varying parameters.
        n_samples : int, optional
            The number of samples to draw for the optimization process. Default is 100.
        num_iter : int, optional
            The number of iterations for the optimization process. Default is 10.
        num_top_candidates : int, optional
            The number of top candidates to retain after each iteration. Default is 1.
        do_local_opt : bool, optional
            Whether to perform local optimization after the global search. Default is True.
        local_method : str, optional
            The optimization method to use for local optimization. Default is 'L-BFGS-B'.
        shrink_rate : float, optional
            The rate at which to shrink the search space during optimization. Default is 0.5.
        parallel : bool, optional
            Whether to run the optimization in parallel. Default is False.
        local_parallel : bool, optional
            Whether to run local optimization in parallel. Default is False.
        verbose : bool, optional
            If True, enables verbose output during optimization. Default is False.
        verbose_plot : bool, optional
            If True, enables plotting of the optimization process. Default is False.
        seed : int, optional
            Random seed for reproducibility. Default is None.
        fixed_params : dict, optional
            A dictionary of parameters that should remain fixed during optimization. 
            Keys should match those in `param_bounds`. Default is None.
        Raises:
        -------
        ValueError
            If no parameters are free for optimization, if the initial guess does not match 
            fixed values, or if the initial guess is out of bounds.
        """

        self.ode_func = ode_func
        self.error_func = error_func
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.verbose_plot = verbose_plot
        self.n_samples = n_samples
        self.num_iter = num_iter
        self.num_top_candidates = num_top_candidates
        self.do_local_opt = do_local_opt
        self.local_method = local_method
        self.shrink_rate = shrink_rate
        self.parallel = parallel
        self.local_parallel = local_parallel
        self.top_candidates_per_iter = []
        self.history = []

        # Handle fixed parameters directly from bounds if provided as scalar
        self.param_bounds = {}
        self.fixed_params = fixed_params.copy() if fixed_params else {}
        for key, bound in param_bounds.items():
            if isinstance(bound, (int, float)):
                self.fixed_params[key] = bound
            else:
                self.param_bounds[key] = bound

        self.varying_params = list(self.param_bounds.keys())
        if not self.varying_params:
            raise ValueError("At least one parameter must be free (not fixed) for optimization.")

        # Handle initial guess
        if initial_guess is not None:
            self.initial_guess = initial_guess.copy()
            for key, val in self.initial_guess.items():
                if key in self.fixed_params:
                    if val != self.fixed_params[key]:
                        raise ValueError(f"Initial guess for '{key}' = {val} does not match fixed value {self.fixed_params[key]}")
                elif key in self.param_bounds:
                    low, high = self.param_bounds[key]
                    if not (low <= val <= high):
                        raise ValueError(f"Initial guess for '{key}' = {val} is out of bounds ({low}, {high})")
                else:
                    raise ValueError(f"Unknown parameter '{key}' in initial guess.")
        else:
            # Midpoint guess for varying, use fixed values for fixed
            self.initial_guess = {
                k: (v[0] + v[1]) / 2 for k, v in self.param_bounds.items()
            }
            self.initial_guess.update(self.fixed_params)

        self.best_params = self.initial_guess.copy()
        self.best_error = float('inf')

    def get_top_candidates_history(self):
        """
        Returns the history of top candidates from each optimization iteration.
        This method retrieves the stored history of top candidates found during each
        iteration of the optimization process. Each entry in the history corresponds to
        a specific iteration and contains the best parameter sets along with their
        associated error values.
        Returns
        -------
        list
            A list of lists, where each inner list contains tuples of (parameter_dict, error_value)
            representing the top candidates for that iteration. The outer list corresponds to the
            iterations performed during optimization.
        Notes
        -----
        - Each inner list contains the top candidates sorted by their error values.
        - The first element of each inner list is the best candidate for that iteration.
        - If no optimization has been performed, this method returns an empty list.
        - The structure allows easy access to the best candidates at each iteration for further analysis.
        """

        return self.top_candidates_per_iter

    def fit(self):

        """
    Fit the model using a global optimization algorithm to minimize ODE function error.
    
    This method implements a sophisticated parameter optimization strategy that combines
    global exploration with local refinement. It uses Latin Hypercube Sampling (LHS) 
    around the best candidates from each iteration, progressively shrinking search regions
    to converge on optimal parameter values. The algorithm maintains diversity by tracking
    multiple top candidates and optionally performs local optimization for fine-tuning.
    
    The optimization process consists of three main phases:
    
    1. **Global Search**: Iteratively samples parameter space using LHS around current
       best candidates, with search regions that shrink over iterations
    2. **Candidate Selection**: Evaluates all samples and retains the top performers
       for the next iteration
    3. **Local Refinement** (optional): Applies gradient-based optimization to polish
       the final results
    
    Algorithm Details
    -----------------
    The method uses an adaptive sampling strategy where:
    
    - Each iteration generates samples around the current top candidates
    - Search regions shrink by a factor of `shrink_rate` each iteration
    - Multiple candidates are maintained to preserve solution diversity
    - Parallel evaluation is supported for computational efficiency
    
    Returns
    -------
    tuple of (dict, float)
        A tuple containing:
        
        - **best_params** (dict): The optimal parameter set found during optimization.
          Contains both varying and fixed parameters as key-value pairs.
        - **best_error** (float): The minimum error value achieved by the best parameters.
          This represents the objective function value at the optimum.
    
    Raises
    ------
    RuntimeError
        If all parameter evaluations fail during any iteration. This typically occurs
        when the ODE function encounters numerical issues or parameter bounds are 
        too restrictive.
    
    Attributes Modified
    -------------------
    The method updates several instance attributes during execution:
    
    - **best_params** (dict): Updated with the optimal parameter set
    - **best_error** (float): Updated with the minimum error found
    - **history** (list): Appends the best error from each iteration
    - **top_candidates_per_iter** (list): Stores top candidates from each iteration
    
    Configuration Parameters
    ------------------------
    The fitting behavior is controlled by several instance attributes:
    
    - **num_iter** (int): Number of optimization iterations to perform
    - **num_top_candidates** (int): Number of best candidates to retain per iteration
    - **n_samples** (int): Number of LHS samples to generate around each candidate
    - **shrink_rate** (float): Rate at which search regions contract (0 < rate < 1)
    - **do_local_opt** (bool): Whether to perform local refinement after global search
    - **parallel** (bool): Enable parallel evaluation of candidate parameters
    - **local_parallel** (bool): Enable parallel local refinement
    - **verbose** (bool): Print detailed progress information
    - **verbose_plot** (bool): Display error history plot after fitting
    
    Notes
    -----
    - The algorithm maintains both varying and fixed parameters. Only varying parameters
      are optimized, while fixed parameters remain constant throughout.
    - Search regions are bounded by the original parameter bounds specified in
      `param_bounds`, ensuring samples never exceed feasible ranges.
    - The shrinking search strategy balances exploration and exploitation, starting
      with broad sampling and gradually focusing on promising regions.
    - Local refinement uses gradient-based methods (specified by `local_method`)
      which can significantly improve convergence for smooth objective functions.
    - Progress tracking through `trange` provides real-time feedback on optimization
      status and estimated completion time.
    
    Examples
    --------
    Basic usage with default settings:
    
    >>> optimizer = ODEOptimizer(
    ...     ode_func=my_ode_solver,
    ...     error_func=my_error_metric,
    ...     param_bounds={'k1': (0.1, 10), 'k2': (0.01, 1)},
    ...     varying_params=['k1', 'k2']
    ... )
    >>> best_params, best_error = optimizer.fit()
    >>> print(f"Optimal parameters: {best_params}")
    >>> print(f"Final error: {best_error:.6f}")
    
    With custom configuration for faster convergence:
    
    >>> optimizer = ODEOptimizer(
    ...     ode_func=my_ode_solver,
    ...     error_func=my_error_metric,
    ...     param_bounds={'k1': (0.1, 10), 'k2': (0.01, 1)},
    ...     varying_params=['k1', 'k2'],
    ...     num_iter=50,
    ...     num_top_candidates=5,
    ...     shrink_rate=0.7,
    ...     do_local_opt=True,
    ...     parallel=True
    ... )
    >>> best_params, best_error = optimizer.fit()
    
    Accessing optimization history:
    
    >>> optimizer.fit()
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(optimizer.history)
    >>> plt.xlabel('Iteration')
    >>> plt.ylabel('Best Error')
    >>> plt.title('Optimization Progress')
    >>> plt.show()
    
    Performance Considerations
    --------------------------
    - **Parallel Evaluation**: Enable `parallel=True` for computationally expensive
      ODE functions. Most effective when `n_samples * num_top_candidates > num_cores`.
    - **Local Optimization**: Set `do_local_opt=True` for smooth objective functions
      where gradient information is available and reliable.
    - **Memory Usage**: Large values of `num_iter` or `num_top_candidates` increase
      memory usage through history storage.
    - **Convergence**: Monitor `history` attribute to detect premature convergence
      or the need for additional iterations.
    
    See Also
    --------
    get_top_candidates_history : Access the complete candidate history
    get_top_candidates_table : Get optimization results as a pandas DataFrame
    plot_error_history : Visualize the optimization progress
    local_refine : The local optimization function used in refinement phase
    lhs_sample : Latin Hypercube Sampling function for parameter generation
    
    References
    ----------
    .. [1] McKay, Michael D., Richard J. Beckman, and William J. Conover. "A comparison of three
            methods for selecting values of input variables in the analysis of output from a computer code." Technometrics 42.1 (2000): 55-61.
    .. [2] Nocedal, J., & Wright, S. (2006). "Numerical optimization".
    """
        top_candidates = [(self.best_params.copy(), float('inf'))]

        for iteration in trange(self.num_iter, desc="Fitting Progress"):
            if self.verbose:
                print(f"\nIteration {iteration + 1}/{self.num_iter}")

            all_sampled = []

            for candidate_params, _ in top_candidates:
                local_bounds = {}
                for key in self.varying_params:
                    full_min, full_max = self.param_bounds[key]
                    center = candidate_params[key]
                    width = (full_max - full_min) * (self.shrink_rate / 2)
                    new_min = max(center - width, full_min)
                    new_max = min(center + width, full_max)
                    local_bounds[key] = (new_min, new_max)

                local_samples = lhs_sample(local_bounds, self.n_samples, seed=self.rng.integers(1e9))

                for sample in local_samples:
                    full_sample = {**sample, **self.fixed_params}
                    all_sampled.append(full_sample)

            def evaluate(param_set):
                try:
                    output = self.ode_func(param_set)
                    err = self.error_func(output)
                    return (param_set, err)
                except Exception as e:
                    if self.verbose:
                        print(f"Evaluation failed for params {param_set}: {e}")
                    return None

            if self.parallel:
                with ProcessPoolExecutor() as executor:
                    results = executor.map(evaluate, all_sampled)
                    evaluated = [res for res in results if res is not None]
            else:
                evaluated = [res for res in map(evaluate, all_sampled) if res is not None]

            if not evaluated:
                raise RuntimeError("All evaluations failed. Check ODE function and parameter ranges.")

            evaluated.sort(key=lambda x: x[1])
            top_candidates = evaluated[:self.num_top_candidates]
            self.top_candidates_per_iter.append(top_candidates.copy())

            if top_candidates[0][1] < self.best_error:
                self.best_params = top_candidates[0][0]
                self.best_error = top_candidates[0][1]

            self.history.append(self.best_error)

            if self.verbose:
                print(f"Best error so far: {self.best_error:.4f}")
                print(f"Best params: {self.best_params}")

        # Local refinement
        if self.do_local_opt:
            bounds_for_local = {k: self.param_bounds[k] for k in self.varying_params}

            if self.local_parallel:
                def local_worker(p):
                    var_params = {k: v for k, v in p[0].items() if k in self.varying_params}
                    
                    refined_param, refined_error = local_refine(
                        var_params,
                        self.ode_func,
                        self.error_func,
                        bounds_for_local,
                        method=self.local_method
                    )
                    full_param = {**refined_param, **self.fixed_params}
                    return (full_param, refined_error)

                with ProcessPoolExecutor() as executor:
                    refined_candidates = list(executor.map(local_worker, top_candidates))
            else:
                refined_candidates = []
                for i, (params, _) in enumerate(top_candidates):
                    var_params = {k: v for k, v in params.items() if k in self.varying_params}
                    print(f"Refining params: {var_params}")
                    refined_param, refined_error = local_refine(
                        var_params,
                        self.ode_func,
                        self.error_func,
                        self.fixed_params,
                        bounds_for_local,
                        method=self.local_method,
                        verbose=self.verbose
                    )
                    #refined_param, refined_error = local_refine(
                    #    var_params,
                    #    self.ode_func,
                    #    self.error_func,
                    #    bounds_for_local,
                    #    method=self.local_method,
                    #    verbose=self.verbose
                    #)

                    full_param = {**refined_param, **self.fixed_params}
                    refined_candidates.append((full_param, refined_error))

            refined_candidates.sort(key=lambda x: x[1])
            self.best_params, self.best_error = refined_candidates[0]

            if self.verbose:
                print("After local refinement:")
                print(f"Best params: {self.best_params}")
                print(f"Best error: {self.best_error:.4f}")

        if self.verbose_plot:
            self.plot_error_history()

        return self.best_params, self.best_error

        
    def best_params(self):
        """
        Retrieve the best parameters found during the optimization process.

        This method returns the best parameters that have been identified 
        by the optimizer. The parameters are typically the result of 
        an optimization algorithm that seeks to minimize or maximize 
        a specific objective function.

        Returns:
            dict: A dictionary containing the best parameters found, 
                  where keys are parameter names and values are the 
                  corresponding optimal values.
        """
        """Returns the best parameters found."""
        return self.best_params
    
    def best_error(self):
        """
        Returns the best error found during the optimization process.

        This method retrieves the value of the best error that has been recorded
        by the optimizer. The best error is typically the lowest error value 
        achieved by the optimization algorithm, indicating the best fit to the 
        data or model being optimized.

        Returns:
            float: The best error value found.
        """
        """Returns the best error found."""
        return self.best_error
    

    def plot_error_history(self, figsize=(6, 4), xlabel='Iteration', ylabel='Best Error', title='Error over Optimization Iterations', fontsize=12):
        """
        Plots the optimization error over iterations.

        This method generates a line plot representing the best error recorded during 
        the optimization process across multiple iterations. It provides a visual 
        representation of how the optimization error changes over time, which can be 
        useful for diagnosing the performance of the optimization algorithm.

        Parameters:
        -----------
        figsize : tuple, optional
            The size of the figure to be created (default is (6, 4)).
        xlabel : str, optional
            The label for the x-axis (default is 'Iteration').
        ylabel : str, optional
            The label for the y-axis (default is 'Best Error').
        title : str, optional
            The title of the plot (default is 'Error over Optimization Iterations').
        fontsize : int, optional
            The font size for the labels and title (default is 12).

        Returns:
        --------
        None

        Raises:
        -------
        None

        Notes:
        ------
        If there is no optimization history available, a message will be printed 
        indicating that there is no data to plot.
        """
        """Plots the optimization error over iterations."""
        if not self.history:
            print("No optimization history to plot.")
            return

        plt.figure(figsize=figsize)
        plt.plot(self.history, marker='o')
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.title(title, fontsize=fontsize)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def summary(self, return_dict=False):
        """
        Summary of the optimizer settings and current best result.
        This method provides a comprehensive overview of the optimizer's configuration and its best-found parameters. It can either print the summary to the console or return it as a dictionary, depending on the value of the `return_dict` parameter.
        return_dict : bool, optional
            If True, the summary is returned as a dictionary. If False (default), the summary is printed to the console.
        Returns
        -------
        dict or None
            Returns a dictionary containing the optimizer settings and results if `return_dict` is True; otherwise, returns None.
        Attributes Included in Summary
        -------------------------------
        - ode_func: The name of the ODE function being optimized.
        - error_func: The name of the error function used for optimization.
        - param_bounds: The bounds for the parameters being optimized.
        - initial_guess: The initial guess for the parameters.
        - n_samples: The number of samples used in the optimization process.
        - num_iter: The number of iterations performed during optimization.
        - num_top_candidates: The number of top candidates to consider.
        - do_local_opt: A flag indicating whether local optimization is performed.
        - local_method: The method used for local optimization.
        - shrink_rate: The rate at which the search space is shrunk.
        - parallel: A flag indicating whether parallel processing is enabled.
        - local_parallel: A flag indicating whether local optimization should be parallelized.
        - verbose: A flag indicating whether verbose output is enabled.
        - verbose_plot: A flag indicating whether verbose plotting is enabled.
        - seed: The random seed used for reproducibility.
        - best_error: The best error found during optimization.
        - best_params: The parameters corresponding to the best error found.
        """
    
        summary_dict = {
            "ode_func": self.ode_func.__name__ if hasattr(self.ode_func, '__name__') else str(self.ode_func),
            "error_func": self.error_func.__name__ if hasattr(self.error_func, '__name__') else str(self.error_func),
            "param_bounds": self.param_bounds,
            "initial_guess": self.initial_guess,
            "n_samples": self.n_samples,
            "num_iter": self.num_iter,
            "num_top_candidates": self.num_top_candidates,
            "do_local_opt": self.do_local_opt,
            "local_method": self.local_method,
            "shrink_rate": self.shrink_rate,
            "parallel": self.parallel,
            "local_parallel": self.local_parallel,
            "verbose": self.verbose,
            "verbose_plot": self.verbose_plot,
            "seed": self.seed,
            "best_error": self.best_error,
            "best_params": self.best_params
        }

        if return_dict:
            return summary_dict

        print("🔍 ODEOptimizer Summary:")
        for k, v in summary_dict.items():
            print(f"  {k}: {v}")


    

    def get_top_candidates_table(self):
        """
    Returns a pandas DataFrame with the top candidates and their errors from all iterations.
    
    This method aggregates the optimization history by collecting the top candidate
    parameter sets and their corresponding error values from each iteration, then
    structures them into a comprehensive DataFrame for analysis and visualization.
    
    The resulting DataFrame provides a complete view of the optimization process,
    showing how candidate solutions evolved across iterations and their relative
    performance rankings.
    
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the top candidates from all optimization iterations.
        Each row represents one candidate solution with the following columns:
        
        - **iteration** (int): The iteration number (1-indexed)
        - **rank** (int): The ranking of the candidate within its iteration (1-indexed, 
          where 1 is the best performing candidate)
        - **error** (float): The error/loss value for this candidate
        - **Additional columns**: All parameter names and their values from the 
          parameter dictionary are flattened into separate columns
    
    Notes
    -----
    - The iteration and rank columns are 1-indexed for better readability
    - Parameter dictionaries are flattened, so each parameter becomes its own column
    - The DataFrame structure allows for easy filtering, sorting, and analysis of 
      optimization progress
    - If no optimization history exists, an empty DataFrame is returned
    
    Examples
    --------
    >>> optimizer = YourOptimizerClass()
    >>> # ... run optimization ...
    >>> df = optimizer.get_top_candidates_table()
    >>> print(df.head())
       iteration  rank     error  param1  param2  param3
    0          1     1  0.125430    0.45    2.1    True
    1          1     2  0.134521    0.52    1.8   False
    2          2     1  0.098234    0.48    2.3    True
    3          2     2  0.112456    0.41    2.0    True
    
    >>> # Filter to see only the best candidate from each iteration
    >>> best_per_iter = df[df['rank'] == 1]
    
    >>> # Analyze parameter evolution over iterations
    >>> import matplotlib.pyplot as plt
    >>> best_per_iter.plot(x='iteration', y='error')
    >>> plt.title('Best Error vs Iteration')
    >>> plt.show()
    
    See Also
    --------
    get_top_candidates_history : Returns the raw history data used by this method
    
    Raises
    ------
    AttributeError
        If the optimizer instance doesn't have the required history tracking methods
    KeyError
        If the parameter dictionaries contain inconsistent keys across iterations
    """
        records = []
        history = self.get_top_candidates_history()
        for iter_idx, candidates in enumerate(history):
            for rank, (params, error) in enumerate(candidates):
                row = {
                    'iteration': iter_idx + 1,
                    'rank': rank + 1,
                    'error': error,
                }
                row.update(params)  # flatten the parameter dict into columns
                records.append(row)

        return pd.DataFrame(records)
        
