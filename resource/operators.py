import numpy as np

def basicGVD(dispParameter,centFrequency,stepsize):
    #evaluates value of freq-domain GVD operator
    #first get value of the basic GVD operator - basic NLSE
    exponent = 2*(np.pi**2)*(dispParameter/2)*(centFrequency**2)*stepsize
    exponent = 0 + (exponent*1j)
    return np.exp(exponent)