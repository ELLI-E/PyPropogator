#file for testing new code
import numpy as np
from matplotlib import pyplot as plt
import resource.pulseshape as pulse
import resource.operators as op
import math


#testing GVD script
b2 = -5 #in ps^2m^-1
nu = 300
stepsize = 5

t = np.linspace(-10,10,1024)
gaussPulseInitial = pulse.GaussianPulse(t,1)
#apply gvd to test pulse once
gaussPulse = op.resolveGVD(b2,stepsize,gaussPulseInitial)

for i in range(1000):
    gaussPulse = op.resolveGVD(b2,stepsize,gaussPulse)


plt.plot(t,np.square(gaussPulseInitial),label = "input")
plt.plot(t,np.square(np.abs(gaussPulse)),label = "after GVD")
plt.legend(loc="best")
plt.ylabel(r"$I(T)$")
plt.xlabel(r"$T$")
plt.show()




"""t = np.linspace(-30,30,1024)
gaussianPulse1 = pulse.GaussianPulse(t,0.5)
FTGaussianPulse1 = np.fft.fft(gaussianPulse1,norm="ortho")
frequency = np.fft.fftfreq(1024)
plt.plot(frequency,np.absolute(FTGaussianPulse1))
plt.show()"""