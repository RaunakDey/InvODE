

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from .util_optimizer import optimization, lhs_sample
import scipy.io

def load_matlab_data(file_path):
    data = scipy.io.loadmat(file_path)
    return data

