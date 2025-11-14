import numpy as np

def basicGVD(dispParameter,centFrequency,stepsize):
    #evaluates value of freq-domain GVD operator
    #first get value of the basic GVD operator - basic NLSE
    exponent = 2*(np.pi**2)*(dispParameter)*(centFrequency**2)*(stepsize/2)
    exponent = 0 + (exponent*1j)
    return np.exp(exponent)

def resolveBasicGVD(dispParameter,stepsize,pulseShape,samplingRate = 1):
    #applies gvd operator in frequency domain and reverses product fourier transform to return to time domain
    pulseShapeFT = np.fft.fft(pulseShape)
    #get frequency space
    frequencies = np.fft.fftfreq(len(pulseShape),1/samplingRate)
    for i,frequency in enumerate(frequencies):
        #multiply each frequency element
        pulseShapeFT[i] = np.multiply(pulseShapeFT[i],basicGVD(dispParameter,frequency,stepsize))
        #print(basicGVD(dispParameter,frequency,stepsize))
    return np.fft.ifft(pulseShapeFT)
def SymSplitStepNL(gamma,pulse,stepsize):
    exponent = gamma*np.square(np.abs(pulse))*stepsize
    exponent = 0 + exponent*1j
    return np.exp(exponent)
def BasicRKNL(gamma,pulse,stepsize):
    return gamma*np.square(np.abs(pulse))*stepsize*(1j)

def BasicRK4IP(pulse,b2,gamma,stepSize,samplingRate):
    #get k1-k4 parts
    PulseIP = resolveBasicGVD(b2,stepSize,pulse,samplingRate)
    k1 = resolveBasicGVD(b2,stepSize,np.multiply(pulse,BasicRKNL(gamma,pulse,stepSize)))
    k2 = np.multiply(BasicRKNL(gamma,np.add(PulseIP,np.divide(k1,2)),stepSize),np.add(PulseIP,np.divide(k1,2)))
    k3 = np.multiply(BasicRKNL(gamma,np.add(PulseIP,np.divide(k2,2)),stepSize),np.add(PulseIP,np.divide(k2,2)))
    k4 = np.multiply(BasicRKNL(gamma,resolveBasicGVD(b2,stepSize,np.add(PulseIP,k3),samplingRate),stepSize),resolveBasicGVD(b2,stepSize,np.add(PulseIP,k3),samplingRate))
    #sum parts together
    s1 = np.add(PulseIP,np.divide(k1,6))
    s2 = np.add(s1,np.divide(k2,3))
    s3 = np.add(s2,np.divide(k3,3))
    #final step
    step = np.add(np.divide(k4,6),resolveBasicGVD(b2,stepSize,s3,samplingRate))
    return step