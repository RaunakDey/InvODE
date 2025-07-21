from util_optimizer import lhs_sample

param_bounds = {
    'phi': (1e-10, 1e-8),
    'beta': (50, 300),
    'd': (0.01, 0.1)
}

samples = lhs_sample(param_bounds, n_samples=10, seed=42)

for i, s in enumerate(samples):
    print(f"Sample {i+1}: {s}")