#file contains pulse shapes
import numpy as np

def GaussianPulse(t,T0):
    return np.exp(-np.divide(np.square(t),T0**2))