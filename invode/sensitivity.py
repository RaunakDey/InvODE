# ode_fit/optimizer.py
from scipy.optimize import minimize
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from .sampling import lhs_sample
from .utils import local_refine
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import trange  


class ODESensitivity:
    def __init__(
        self,
        ode_func,
        error_func,
        
    ):
        raise NotImplementedError("This class is not implemented yet. Please use ODEOptimizer for optimization tasks.")