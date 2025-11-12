import numpy as np

def basicGVD(dispParameter,centFrequency,stepsize):
    #evaluates value of freq-domain GVD operator
    #first get value of the basic GVD operator - basic NLSE
    exponent = 2*(np.pi**2)*(dispParameter/2)*(centFrequency**2)*(stepsize/2)
    exponent = 0 + (exponent*1j)
    return np.exp(exponent)

def resolveBasicGVD(dispParameter,stepsize,pulseShape):
    #applies gvd operator in frequency domain and reverses product fourier transform to return to time domain
    pulseShapeFT = np.fft.fft(pulseShape)
    #get frequency space
    frequencies = np.fft.fftfreq(len(pulseShape))
    for i,frequency in enumerate(frequencies):
        #multiply each frequency element
        pulseShapeFT[i] = np.multiply(pulseShapeFT[i],basicGVD(dispParameter,frequency,stepsize))
        #print(basicGVD(dispParameter,frequency,stepsize))
    return np.fft.ifft(pulseShapeFT)
def SymSplitStepNL(gamma,pulse,stepsize):
    exponent = gamma*np.square(np.abs(pulse))*stepsize
    exponent = 0 + exponent*1j
    return np.exp(exponent)
