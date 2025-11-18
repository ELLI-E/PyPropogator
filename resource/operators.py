import numpy as np
from math import factorial
from copy import deepcopy
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

def GeneralGVD(dispList,frequency,attenuation,stepsize):
    #sum over higher order frequency dispersion parameters
    summed = 0
    for i,dispersion in enumerate(dispList):
        summed += (dispersion/factorial(i+2))*((2*np.pi*frequency)**2)
    exponent = (-attenuation/2) + (summed*1j)
    return np.exp(np.multiply(exponent,stepsize/2))

def SymSplitStepNL(gamma,pulse,stepsize):
    exponent = gamma*np.square(np.abs(pulse))*stepsize
    exponent = 0 + exponent*1j
    return np.exp(exponent)

def BasicRKNL(gamma,pulse,stepsize):
    return gamma*np.square(np.abs(pulse))*stepsize*(1j)

def BasicRK4IP(pulse,b2,gamma,stepSize,samplingRate):
    #get k1-k4 parts
    PulseIP = resolveBasicGVD(b2,stepSize,pulse,samplingRate)
    k1 = resolveBasicGVD(b2,stepSize,np.multiply(pulse,BasicRKNL(gamma,pulse,stepSize)),samplingRate)
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

def RamanResponseIntegral(ramanresponse,pulseIn):
    """
    ramanfunction - values of raman function with the same spacing between points in time as pulse
    pulse - shape of pulse - NOTE requires plenty of temporal space for the pulse to be "shifted" while performing the integral, at least a picosecond
    """
    #this whole function can probably be made redundant through a Fourier Transform - will look into the specifics at a later date
    #for safety make pulse a deepcopy of pulseIn
    pulse = deepcopy(pulseIn)
    ramanPulse = np.zeros(len(pulse))
    for i,raman in enumerate(ramanresponse):
        ramanPulse = np.add(ramanPulse,np.multiply(raman,np.square(pulse)))
        pulse = np.roll(pulse,1) #shift pulse by 1 time space
        pulse[0] #to prevent looping in array, make the leftmost value 0
    return ramanPulse

def GeneralNL(gamma,ramanCurve,ramanFraction,centFrequency,pulseIn,samplingRate=1):
    #will use similar method to ResolveBasicGVD to eliminate differential part
    #first resolve RHS part 
    rhs1 = np.multiply(pulseIn,np.square(np.abs(pulseIn)))
    rhs1 = np.multiply(rhs1,(1-ramanFraction))
    rhs2 = np.multiply(np.multiply(pulseIn,ramanFraction),RamanResponseIntegral(ramanCurve,pulseIn))
    rhs = np.add(rhs1,rhs2)
    rhsfft = np.fft.fft(rhs)
    fftfrequency = np.fft.fftfreq(len(pulseIn),samplingRate)
    for i,frequency in enumerate(fftfrequency):
        rhsfft[i] = np.multiply(rhsfft[i],2*np.pi*frequency)
    lhs1 = np.fft.ifft(rhsfft)
    lhs1 = np.multiply(lhs1,np.divide(1j,np.multiply(pulseIn,centFrequency)))
    lhs2 = np.divide(rhs,pulseIn)
    lhs = np.add(lhs1,lhs2)
    lhs = np.multiply(1j * gamma,lhs)
    return lhs
