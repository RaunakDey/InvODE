import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def sine(y, t, A, omega, delta):
    dy_dt = A * omega * np.sin(omega * t + delta)
    return dy_dt

def simulate_model(params):
    y0 = 1.0  # Initial condition
    A = params['A']
    omega = params['omega']
    delta = params['delta']
    sol = odeint(sine, y0, time, args=(A, omega, delta))
    return sol.flatten()


if __name__ == "__main__":
    time = np.linspace(0, 10, 50)  # Time array
    # Example usage
    A = 1.0      # Amplitude
    omega = 2.0  # Angular frequency
    delta = 0.0  # Phase shift
    t = np.linspace(0, 10, 100)  # Time array
    y = np.zeros_like(t)          # Initialize y array
    params = {'A': A, 'omega': omega, 'delta': delta}
    y = simulate_model(params)
    y = y + np.random.normal(0, 0.1, len(y))  # Adding noise to the output
    print("Simulation completed.\n")
    data = { "Time": time,
    "y": y,}
    print(data)
    # plotting the results

    plt.plot(time, y, '.',label='Simulated Sine Wave')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Sine Wave Simulation')
    plt.legend()
    plt.show()


    