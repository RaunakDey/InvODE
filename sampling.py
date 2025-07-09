# sampling.py
import numpy as np
from scipy.stats import qmc

def lhs_sample(bounds, n, seed=None):
    sampler = qmc.LatinHypercube(d=len(bounds), seed=seed)
    sample = sampler.random(n)
    lower = np.array([b[0] for b in bounds.values()])
    upper = np.array([b[1] for b in bounds.values()])
    scaled = qmc.scale(sample, lower, upper)
    keys = list(bounds.keys())
    return [dict(zip(keys, row)) for row in scaled]


