"""
ode_fit: A lightweight package for parameter fitting of ODE models
         using Latin Hypercube Sampling and local optimization.
"""

from .optimizer import ODEOptimizer
from .sampling import lhs_sample
from .utils import shrink_bounds, check_bounds

__all__ = ["ODEOptimizer", "lhs_sample", "shrink_bounds", "check_bounds"]

__version__ = "0.1.0"