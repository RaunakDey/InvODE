import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from util_optimizer import naive_optimization, lhs_sample

# === Define ground-truth ODE model ===

def true_model(y, t, a, b):
    return a * y + b

# === Generate synthetic data ===

true_params = {'a': -0.5, 'b': 2.0}
y0 = 1.0
t = np.linspace(0, 5, 100)

# Simulate ground truth
def ode_rhs(y, t, params):
    return true_model(y, t, **params)

true_solution = odeint(ode_rhs, y0, t, args=(true_params,)).flatten()
noisy_data = true_solution + np.random.normal(0, 0.1, size=true_solution.shape)

# === Define simulate_model and error_func ===

def simulate_model(params):
    sol = odeint(ode_rhs, y0, t, args=(params,))
    return sol.flatten()

def mse(output):
    return np.mean((output - noisy_data)**2)

# === Use naive_optimization ===

param_bounds = {
    'a': (-1.5, 0.5),
    'b': (0.0, 4.0)
}

initial_guess = {'a': 0.0, 'b': 1.0}

best_params, best_error = naive_optimization(
    ode_func=simulate_model,
    error_func=mse,
    param_bounds=param_bounds,
    initial_guess=initial_guess,
    n_samples=200,
    num_iter=10,
    verbose=True,
    verbose_plot=True
)

# === Plotting the result ===

best_fit = simulate_model(best_params)

plt.figure(figsize=(8, 5))
plt.plot(t, noisy_data, 'o', label='Noisy data', alpha=0.5)
plt.plot(t, true_solution, label='True model')
plt.plot(t, best_fit, '--', label='Best fit')
plt.legend()
plt.xlabel("Time")
plt.ylabel("y(t)")
plt.title("ODE Fit using Naive Optimization")
plt.show()

print("\nTrue parameters:", true_params)
print("Recovered parameters:", best_params)
print("Final MSE:", best_error)