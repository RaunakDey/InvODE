# ode_fit/optimizer.py
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from .sampling import lhs_sample
from .utils import shrink_bounds, check_bounds, param_dict_to_array, array_to_param_dict

class ODEOptimizer:
    def __init__(self, ode_func, error_func, param_bounds, initial_guess,
                 n_samples=100, num_iter=10, num_top_candidates=1,
                 do_local_opt=True, local_method='L-BFGS-B',
                 shrink_rate=0.5, parallel=True, local_parallel=False,
                 verbose=True, verbose_plot=True, seed=None):

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
        self.history = []

        check_bounds(initial_guess, param_bounds)

    def _evaluate(self, param_set):
        try:
            output = self.ode_func(param_set)
            err = self.error_func(output)
            return (param_set, err)
        except:
            return None

    def _evaluate_all(self, param_sets):
        if self.parallel:
            with ProcessPoolExecutor() as executor:
                results = executor.map(self._evaluate, param_sets)
                return [r for r in results if r is not None]
        else:
            return [self._evaluate(p) for p in param_sets if self._evaluate(p) is not None]

    def _refine_local(self, param_set):
        def obj(theta_array):
            theta_dict = array_to_param_dict(theta_array, self.param_bounds)
            return self.error_func(self.ode_func(theta_dict))

        x0 = param_dict_to_array(param_set, self.param_bounds)
        bounds = list(self.param_bounds.values())
        res = minimize(obj, x0, method=self.local_method, bounds=bounds)
        return array_to_param_dict(res.x, self.param_bounds), res.fun

    def _refine_all_locals(self, top_candidates):
        if self.local_parallel:
            with ProcessPoolExecutor() as executor:
                results = executor.map(self._refine_local, [p[0] for p in top_candidates])
                return list(zip(results, [p[1] for p in top_candidates]))
        else:
            return [(self._refine_local(p[0]), p[1]) for p in top_candidates]

    def fit(self):
        best_candidates = [(self.initial_guess, float('inf'))]
        all_errors = []

        for it in range(self.num_iter):
            if self.verbose:
                print(f"Iteration {it + 1}/{self.num_iter}")

            sample_pool = []

            for param_center, _ in best_candidates:
                bounds_local = shrink_bounds(param_center, self.param_bounds, self.shrink_rate)
                samples = lhs_sample(bounds_local, self.n_samples, seed=self.seed)
                sample_pool.extend(samples)

            evaluated = self._evaluate_all(sample_pool)
            evaluated.sort(key=lambda x: x[1])

            best_candidates = evaluated[:self.num_top_candidates]
            all_errors.append([e[1] for e in best_candidates])

        if self.do_local_opt:
            if self.verbose:
                print("Running local optimization on top candidates...")
            refined = [self._refine_local(p[0]) for p in best_candidates]
            best_candidates = sorted(refined, key=lambda x: x[1])

        self.best_params, self.best_error = best_candidates[0]
        self.history = all_errors

    def get_best(self):
        return self.best_params, self.best_error

    def plot_error_history(self):
        if not self.history or not self.verbose_plot:
            return
        avg_errors = [np.mean(e) for e in self.history]
        plt.plot(avg_errors, marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Average Error (Top Candidates)")
        plt.title("Optimization Error History")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        