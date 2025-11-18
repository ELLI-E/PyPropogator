#raman function generation used in code
import numpy as np
def BlowWoodResponse(maxTime,timeSteps,t1,t2):
    #NOTE: Sampling rates between raman function and pulse MUST match
    t = np.linspace(0,maxTime,timeSteps)
    response = ((t1**2)+(t2**2))/(t1*(t2**2))
    response = np.multiply(response,np.exp(-(t/t2)))
    response = np.multiply(response,np.sin((t/t1)))
    return t,response
