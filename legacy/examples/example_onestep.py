import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os
import sys
import scipy.io
from concurrent.futures import ProcessPoolExecutor

# Get path 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from ode_fit_pointwise import optimization, lhs_sample
from ode_fit_pointwise import load_matlab_data



file_path = './../../sample_data/HS6_13-15_2024.mat' 
free_phages = load_matlab_data(file_path)['free_phages']
S0 = np.mean(load_matlab_data(file_path)['S0_replicates'])
V0 = np.mean(load_matlab_data(file_path)['V0_replicates'])
time = load_matlab_data(file_path)['time_free_phages'].flatten()/60  # Convert time to hours

# Initial guess for parameters ===
initial_guess = {
    'r': 0.25,
    'phi': 4.5e-08 ,
    'beta':  256,
    'tau': 2,
    'NE': 180
}


# Initial conditions ===
y0 = np.zeros((initial_guess['NE']+3,))
y0[0] = S0
y0[-1] = V0



def onstep(y, t, params):

    phi = params['phi']
    beta = params['beta']
    tau = params['tau']
    r = params['r']
    NE = params['NE']

    S = y[0]
    E_mat = y[1:NE+1]
    I = y[NE+1]
    V = y[NE+2]

    etaeff = ((NE+1)/tau)

    dotS = r*S - phi*V*S
    dotE1 = phi*S*V - etaeff * E_mat[0]
    
    if NE > 1:
        dotE_mat = np.zeros(NE-1)
        dotE_mat[:] = etaeff * E_mat[0:-1] - etaeff * E_mat[1:]

    dotI = etaeff * (E_mat[-1] - I)
    dotV = beta * etaeff * I - V * phi * (S + I + np.sum(E_mat))

    # Build the full derivative vector (same length as y)
    dydt = np.zeros_like(y)
    dydt[0] = dotS
    dydt[1:NE+1] = dotE1 if NE == 1 else np.concatenate([[dotE1], dotE_mat])
    dydt[NE+1] = dotI
    dydt[NE+2] = dotV

    return dydt



def simulate_model(params):
    # Initial conditions ===
    params['NE'] = int(params['NE'])
    y0 = np.zeros((params['NE']+3,))
    y0[0] = S0
    y0[-1] = V0


    ## dilution step
    time_dil = np.linspace(0, 0.25, 200)  # Short time for dilution step
    sol_dil = odeint(onstep, y0, time_dil, args=(params,))
    y0_dil = sol_dil[-1, :]/100 # Use the last state as the new initial condition
    ## main simulation
    sol = odeint(onstep, y0_dil, time, args=(params,))
    phage_solution = sol[:,-1]
    return phage_solution 


def mse(model_output):
    target = np.mean(free_phages, axis=1)
    if len(model_output) != len(target):
        raise ValueError("Length mismatch between model output and data")
    return np.mean((np.log10(model_output) - np.log10(target)) ** 2)




free_phages_sol = simulate_model(initial_guess)





# === Run optimization ===
if __name__ == '__main__':
    best_params, best_error = optimization(
        ode_func=simulate_model,
        error_func=mse,
        param_bounds={
            'r': (0.1, 0.5),
            'phi': (1e-8, 1e-7),
            'beta': (100, 500),
            'tau': (1, 5),
            'NE': (100, 200)
        },
        initial_guess=initial_guess,
        n_samples=300,
        num_iter=10,
        verbose=True,
        verbose_plot=True,
        do_local_opt=True,
        local_method='L-BFGS-B',
        num_top_candidates=2,
        show_final_candidates=True,
        parallel=False,
        local_parallel=False
    )

phages_fit = simulate_model(best_params)


# Plot the initial guess
plt.figure(figsize=(10, 6))
plt.plot(time, free_phages, label='Measured Free Phages', color='blue', marker='o', markersize=3)
plt.plot(time, free_phages_sol, label='Initial Guess', linestyle='--', color='orange')
plt.plot(time, phages_fit, label='Optimized Fit', color='red')
plt.xlabel('Time (hours)')
plt.ylabel('Free Phages')
plt.yscale('log')
plt.title('Free Phages Over Time')
plt.legend()
plt.show()
