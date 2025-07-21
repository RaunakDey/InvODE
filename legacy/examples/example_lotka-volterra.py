import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os
import sys
import scipy.io

# Get path to MCMCwithODEs_primer (3 levels up)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from ode_fit_pointwise import optimization, lhs_sample
from ode_fit_pointwise import load_matlab_data


# === Define the true model ===

def lotka_volterra(z, t, params):
    x, y = z
    alpha = params['alpha']
    beta = params['beta']
    delta = params['delta']
    gamma = params['gamma']
    
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# === Generate synthetic data ===

true_params = {
    'alpha': 1.0,    # prey growth rate
    'beta': 0.1,     # predation rate
    'delta': 0.075,  # predator growth per prey eaten
    'gamma': 1.5     # predator death rate
}

t = np.linspace(0, 20, 200)
z0 = [40, 9]  # initial population: 40 prey, 9 predators

true_sol = odeint(lotka_volterra, z0, t, args=(true_params,))
noisy_data = true_sol + np.random.normal(0, 1.0, true_sol.shape)

# === Define simulate_model and error_func ===

def simulate_model(params):
    sol = odeint(lotka_volterra, z0, t, args=(params,))
    return sol  # shape: (N, 2)

def mse(output):
    return np.mean((output - noisy_data)**2)

# === Parameter bounds and initial guess ===
'''
param_bounds = {
    'alpha': (0.5, 1.5),
    'beta': (0.05, 0.2),
    'delta': (0.05, 0.15),
    'gamma': (1.0, 2.0)
}

initial_guess = {
    'alpha': 1.2,
    'beta': 0.15,
    'delta': 0.1,
    'gamma': 1.2
}
'''


param_bounds = {
    'alpha': (0.5, 10),
    'beta': (0.05, 0.9),
    'delta': (0.05, 0.9),
    'gamma': (1.0, 5.0)
}

initial_guess = {
    'alpha': 5,
    'beta': 0.5,
    'delta': 0.7,
    'gamma': 4.0
}


# === Run optimization ===

best_params, best_error = optimization(
    ode_func=simulate_model,
    error_func=mse,
    param_bounds=param_bounds,
    initial_guess=initial_guess,
    n_samples=300,
    num_iter=10,
    verbose=True,
    verbose_plot=True,
    do_local_opt=True,
    local_method='L-BFGS-B',
    num_top_candidates=4,

)

# === Plot results ===

best_fit = simulate_model(best_params)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(t, noisy_data[:, 0], 'o', alpha=0.4, label='Prey data')
plt.plot(t, best_fit[:, 0], label='Prey fit')
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()
plt.title("Prey")

plt.subplot(1, 2, 2)
plt.plot(t, noisy_data[:, 1], 'o', alpha=0.4, label='Predator data')
plt.plot(t, best_fit[:, 1], label='Predator fit')
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()
plt.title("Predator")

plt.suptitle("Lotka–Volterra Fit using Naive Optimization")
plt.tight_layout()
plt.show()

print("\nTrue parameters:", true_params)
print("Recovered parameters:", best_params)
print("Final MSE:", best_error)