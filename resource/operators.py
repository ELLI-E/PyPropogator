import numpy as np
from math import factorial
from copy import deepcopy

"""
This file contains operators and the functions used to resolve them. This includes the symmetric split step and
Runge-Kutta step functions that are called in a loop to iterate over an input pulse. All operations on numpy arrays
use numpy operators, i.e. np.add rather than + to avoid errors and ambiguity
"""
def round(min,input):
    #attempting to avoid overflow errors with very small values - i.e. 1e-255 appearing 
    array = deepcopy(input)
    for i,element in enumerate(array):
        if np.abs(element) <= min:
            array[i] = 0
    return array
        
def basicGVD(dispParameter,centFrequency,stepsize):
    #OBSOLETE - left in file for reference only
    #evaluates value of freq-domain GVD operator
    #first get value of the basic GVD operator - basic NLSE
    exponent = 2*(np.pi**2)*(dispParameter)*(centFrequency**2)*(stepsize/2)
    exponent = 0 + (exponent*1j)
    return np.exp(exponent)

def GeneralGVD(dispList,frequency,attenuation,stepsize):
    #sum over higher order frequency dispersion parameters
    summed = 0
    for i,dispersion in enumerate(dispList):
        summed += (dispersion/factorial(i+2))*((2*np.pi*frequency)**(i+2))*(1j**((2*(i+2))-1))
    exponent = (-attenuation/2) - (summed)
    return np.exp(np.multiply(exponent,stepsize/2))

def SymSplitStepNL(gamma,pulse,stepsize):
    #OBSOLETE - left in file for reference only
    exponent = gamma*np.square(np.abs(pulse))*stepsize
    exponent = 0 + exponent*1j
    return np.exp(exponent)

def BasicRKNL(gamma,pulse,stepsize):
    return gamma*np.square(np.abs(pulse))*stepsize*(1j)

def GeneralNL(gamma,ramanCurve,ramanFraction,centFrequency,pulseIn,samplingRate=1):
    #will use similar method to ResolveBasicGVD to eliminate differential part
    #first resolve RHS part 
    #centfrequency - use central angular frequency if pulse intensity is NOT normalised - i.e. A(z,t) = sqrt(P_0)e^(-alpha z/2)U(z,t)
    #if using U(z,t), instead take centFrequency to be 1/s, or central angular frequency multiplied by the pulse duration, T0
    rhs1 = np.multiply(pulseIn,np.multiply(pulseIn,np.conjugate(pulseIn)))
    rhs1 = np.multiply(rhs1,(1-ramanFraction))
    rhs2 = np.multiply(np.multiply(pulseIn,ramanFraction),RamanResponseConvolution(ramanCurve,np.square(pulseIn)))
    rhs = np.add(rhs1,rhs2)
    rhsfft = np.fft.fft(rhs)
    fftfrequency = np.fft.fftfreq(len(pulseIn),samplingRate)
    for i,frequency in enumerate(fftfrequency):
        rhsfft[i] = np.multiply(rhsfft[i],2j*np.pi*frequency)
    lhs1 = np.fft.ifft(rhsfft)
    #need to make sure that we only divide by values that are not zero
    lhs1 = np.multiply(lhs1,np.divide(1,centFrequency))
    #we ignore elements where pulseIn is zero - we have to divide the others to avoid overflow but zero elements will be multiplied by zero later
    lhs2 = np.multiply(rhs,1)
    #before dividing anything, round
    pulseIn = round(1e-6,pulseIn)
    for i,inputPulseValue in enumerate(pulseIn):
        #skip over elements of value 0
        if inputPulseValue != 0:
            lhs1[i] = np.divide(lhs1[i],inputPulseValue)
            lhs2[i] = np.divide(rhs[i],inputPulseValue)

    lhs = np.add(lhs1,lhs2)
    lhs = np.multiply(1j * gamma,lhs)
    return lhs

def resolveBasicGVD(dispParameter,stepsize,pulseShape,samplingRate = 1):
    #OBSOLETE - left in file for reference only
    #applies gvd operator in frequency domain and reverses product fourier transform to return to time domain
    pulseShapeFT = np.fft.fft(pulseShape)
    #get frequency space
    frequencies = np.fft.fftfreq(len(pulseShape),1/samplingRate)
    for i,frequency in enumerate(frequencies):
        #multiply each frequency element
        pulseShapeFT[i] = np.multiply(pulseShapeFT[i],basicGVD(dispParameter,frequency,stepsize))
        #print(basicGVD(dispParameter,frequency,stepsize))
    return np.fft.ifft(pulseShapeFT)

def ResolveGeneralGVD(dispList,attenuation,stepSize,pulseShape,samplingRate):
    pulseShapeFT = np.fft.fft(pulseShape)
    frequencies = np.fft.fftfreq(len(pulseShape),1/samplingRate)
    for i,frequency in enumerate(frequencies):
        pulseShapeFT[i] = np.multiply(pulseShapeFT[i],GeneralGVD(dispList,frequency,attenuation,stepSize))
    return np.fft.ifft(pulseShapeFT)

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
        pulse[0] = 0 #to prevent looping in array, make the leftmost value 0
    return ramanPulse
def RamanResponseConvolution(ramanresponse,pulseIn):
    ramanpulse = np.convolve(pulseIn,ramanresponse,"same")
    #compensate for spurious gains
    ratio = np.sum(np.abs(ramanpulse))/np.sum(np.abs(pulseIn))
    ramanpulse = np.divide(ramanpulse,ratio)
    return ramanpulse
def BasicRK4IP(pulse,b2,gamma,stepSize,samplingRate):
    #OBSOLETE - left in file for reference only
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

def GeneralGVDRK4IP(dispList,attenuation,gamma,stepSize,samplingRate,pulseIn):
    #using general GVD, but basic nonlinear parameter as in BasicRKNL
    PulseIP = ResolveGeneralGVD(dispList,attenuation,stepSize,pulseIn,samplingRate)
    k1 = ResolveGeneralGVD(dispList,attenuation,stepSize,np.multiply(pulseIn,BasicRKNL(gamma,pulseIn,stepSize)),samplingRate)
    k2 = np.multiply(BasicRKNL(gamma,np.add(PulseIP,np.divide(k1,2)),stepSize),np.add(PulseIP,np.divide(k1,2)))
    k3 = np.multiply(BasicRKNL(gamma,np.add(PulseIP,np.divide(k2,2)),stepSize),np.add(PulseIP,np.divide(k2,2)))
    k4 = np.multiply(BasicRKNL(gamma,ResolveGeneralGVD(dispList,attenuation,stepSize,np.add(PulseIP,k3),samplingRate),stepSize),ResolveGeneralGVD(dispList,attenuation,stepSize,np.add(PulseIP,k3),samplingRate))
    #sum parts together
    s1 = np.add(PulseIP,np.divide(k1,6))
    s2 = np.add(s1,np.divide(k2,3))
    s3 = np.add(s2,np.divide(k3,3))
    #final step
    step = np.add(np.divide(k4,6),ResolveGeneralGVD(dispList,attenuation,stepSize,s3,samplingRate))
    return step

def GNLSERK4IP(dispList,attenuation,gamma,ramanCurve,ramanFraction,centFrequency,stepSize,samplingRate,pulseIn):
    PulseIP = ResolveGeneralGVD(dispList,attenuation,stepSize,pulseIn,samplingRate)
    k1 = ResolveGeneralGVD(dispList,attenuation,stepSize,np.multiply(pulseIn,np.multiply(stepSize,GeneralNL(gamma,ramanCurve,ramanFraction,centFrequency,pulseIn,samplingRate))),samplingRate)
    k2 = np.multiply(GeneralNL(gamma,ramanCurve,ramanFraction,centFrequency,np.add(PulseIP,np.divide(k1,2)),samplingRate),np.add(PulseIP,np.divide(k1,2)))
    k2 = np.multiply(k2,stepSize)
    k3 = np.multiply(GeneralNL(gamma,ramanCurve,ramanFraction,centFrequency,np.add(PulseIP,np.divide(k2,2)),samplingRate),np.add(PulseIP,np.divide(k2,2)))
    k3 = np.multiply(k3,stepSize)
    k4 = np.multiply(GeneralNL(gamma,ramanCurve,ramanFraction,centFrequency,ResolveGeneralGVD(dispList,attenuation,stepSize,np.add(PulseIP,k3),samplingRate),samplingRate),ResolveGeneralGVD(dispList,attenuation,stepSize,np.add(PulseIP,k3),samplingRate))
    k4 = np.multiply(k4,stepSize)
    #sum parts together
    s1 = np.add(PulseIP,np.divide(k1,6))
    s2 = np.add(s1,np.divide(k2,3))
    s3 = np.add(s2,np.divide(k3,3))
    #final step
    step = np.add(np.divide(k4,6),ResolveGeneralGVD(dispList,attenuation,stepSize,s3,samplingRate))
    return step