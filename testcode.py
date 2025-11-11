#file for testing new code
import numpy as np
from matplotlib import pyplot as plt
import resource.pulseshape as pulse
import math
def basicGVD(dispParameter,centFrequency,stepsize):
    #first get value of the basic GVD operator - basic NLSE
    exponent = 2*(np.pi**2)*dispParameter*(centFrequency**2)*stepsize
    exponent = 0 + (exponent*1j)
    return np.exp(exponent)

#testing GVD script
b2 = 1 #in ps^2m^-1
nu = 300



"""t = np.linspace(-30,30,1024)
gaussianPulse1 = pulse.GaussianPulse(t,0.5)
FTGaussianPulse1 = np.fft.fft(gaussianPulse1,norm="ortho")
frequency = np.fft.fftfreq(1024)
plt.plot(frequency,np.absolute(FTGaussianPulse1))
plt.show()"""