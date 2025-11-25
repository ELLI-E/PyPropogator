#raman function generation used in code
import numpy as np
def BlowWoodResponse(maxTime,timeSteps,t1=12.2,t2=32):
    #NOTE: Sampling rates between raman function and pulse MUST match
    t = np.linspace(0,maxTime,timeSteps)
    response = ((t1**2)+(t2**2))/(t1*(t2**2))
    response = np.multiply(response,np.exp(-(t/t2)))
    response = np.multiply(response,np.sin((t/t1)))
    return t,response

def BlowWoodResponse2(time,t1=12.2,t2=32):
    response = ((t1**2)+(t2**2))/(t1*(t2**2))
    response = np.multiply(response,np.exp(-(time/t2)))
    response = np.multiply(response,np.sin((time/t1)))
    #same as before, except now we make all values of response for negative t equal to 0
    response = [r if time[i] >= 0 else 0 for i,r in enumerate(response)]
    return response 