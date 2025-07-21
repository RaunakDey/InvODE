# ode_fit/optimizer.py
from scipy.optimize import minimize
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from .sampling import lhs_sample
from .utils import local_refine
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import trange  


from concurrent.futures import ProcessPoolExecutor
from tqdm import trange
import numpy as np
from scipy.optimize import minimize


from concurrent.futures import ProcessPoolExecutor
from tqdm import trange
import numpy as np

class ODEOptimizer:
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
        return self.top_candidates_per_iter

    def fit(self):
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
        """Returns the best parameters found."""
        return self.best_params
    
    def best_error(self):
        """Returns the best error found."""
        return self.best_error
    

    def plot_error_history(self, figsize=(6, 4), xlabel='Iteration', ylabel='Best Error', title='Error over Optimization Iterations', fontsize=12):
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
        Prints or returns a summary of the optimizer settings and current best result.
        
        Parameters
        ----------
        return_dict : bool
            If True, returns the summary as a dictionary instead of printing.
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
        Returns a pandas DataFrame with the top candidates and their errors
        from all iterations.
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
        
