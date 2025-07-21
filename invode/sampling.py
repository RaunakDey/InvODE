# sampling.p

import numpy as np
from scipy.stats import qmc

def lhs_sample(param_bounds, n_samples, seed=None):
    """
    Generate Latin Hypercube Samples within specified bounds for each parameter.
    """
    keys = list(param_bounds.keys())
    bounds = np.array([param_bounds[k] for k in keys])  # shape: (n_params, 2)

    sampler = qmc.LatinHypercube(d=len(keys), seed=seed)
    unit_samples = sampler.random(n=n_samples)
    scaled_samples = qmc.scale(unit_samples, bounds[:, 0], bounds[:, 1])
    return [dict(zip(keys, sample)) for sample in scaled_samples]
