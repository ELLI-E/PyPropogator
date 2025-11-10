#file for testing new code
import numpy as np
from matplotlib import pyplot as plt
import resource.pulseshape as pulse

t = np.linspace(-10,10,100)
print(pulse.GaussianPulse(t,5))
plt.plot(t,pulse.GaussianPulse(t,5))
plt.show()