# ode_fit/optimizer.py
from scipy.optimize import minimize
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from .sampling import lhs_sample
from .utils import local_refine
import matplotlib.pyplot as plt
import pandas as pd

class ODEOptimizer:
    def __init__(
        self,
        ode_func,
        error_func,
        param_bounds,
        initial_guess,
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
        seed=None
    ):
        self.ode_func = ode_func
        self.error_func = error_func
        self.param_bounds = param_bounds
        self.initial_guess = initial_guess
        self.n_samples = n_samples
        self.num_iter = num_iter
        self.num_top_candidates = num_top_candidates
        self.do_local_opt = do_local_opt
        self.local_method = local_method
        self.shrink_rate = shrink_rate
        self.parallel = parallel
        self.local_parallel = local_parallel
        self.verbose = verbose
        self.verbose_plot = verbose_plot
        self.seed = seed
        self.top_candidates_per_iter = []  


        self.rng = np.random.default_rng(seed)
        self.best_params = initial_guess.copy()
        self.best_error = float('inf')
        self.history = []  # Error history per iteration

        # Initial validation
        for key, value in initial_guess.items():
            if key not in param_bounds:
                raise ValueError(f"Unknown parameter '{key}' in initial guess.")
            low, high = param_bounds[key]
            if not (low <= value <= high):
                raise ValueError(f"Initial guess for '{key}' = {value} is out of bounds ({low}, {high}).")

    def get_top_candidates_history(self):
        return self.top_candidates_per_iter

    def fit(self):
        top_candidates = [(self.best_params.copy(), float('inf'))]

        for iteration in range(self.num_iter):
            if self.verbose:
                print(f"\nIteration {iteration + 1}/{self.num_iter}")

            all_sampled = []

            for candidate_params, _ in top_candidates:
                local_bounds = {}
                for key in self.param_bounds:
                    full_min, full_max = self.param_bounds[key]
                    width = (full_max - full_min) * (self.shrink_rate / 2)
                    center = candidate_params[key]
                    new_min = max(center - width, full_min)
                    new_max = min(center + width, full_max)
                    local_bounds[key] = (new_min, new_max)

                local_samples = lhs_sample(local_bounds, self.n_samples, seed=self.rng.integers(1e9))
                all_sampled.extend(local_samples)

            def evaluate(param_set):
                try:
                    output = self.ode_func(param_set)
                    err = self.error_func(output)
                    return (param_set, err)
                except Exception:
                    return None

            if self.parallel:
                with ProcessPoolExecutor() as executor:
                    results = executor.map(evaluate, all_sampled)
                    evaluated = [res for res in results if res is not None]
            else:
                evaluated = [res for res in map(evaluate, all_sampled) if res is not None]

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
            


        if self.do_local_opt:
            if self.local_parallel:
                def local_worker(p):
                    return local_refine(
                        p[0], self.ode_func, self.error_func,
                        self.param_bounds, method=self.local_method
                    )

                with ProcessPoolExecutor() as executor:
                    refined_candidates = list(executor.map(local_worker, top_candidates))
            else:
                refined_candidates = []
                for i, (params, _) in enumerate(top_candidates):
                    refined_param, refined_error = local_refine(
                        params, self.ode_func, self.error_func,
                        self.param_bounds, method=self.local_method,
                        verbose=self.verbose
                    )
                    refined_candidates.append((refined_param, refined_error))

            refined_candidates.sort(key=lambda x: x[1])
            self.best_params, self.best_error = refined_candidates[0]

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
        
