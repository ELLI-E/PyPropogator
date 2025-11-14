import operators as op
import numpy as np

def BasicRK4IP(pulse,b2,gamma,stepSize,samplingRate):
    #get k1-k4 parts
    PulseIP = op.resolveBasicGVD(b2,stepSize,pulse,samplingRate)
    k1 = op.resolveBasicGVD(b2,stepSize,np.multiply(pulse,op.BasicRKNL(gamma,pulse,stepSize)))
    k2 = np.multiply(op.BasicRKNL(gamma,np.add(PulseIP,np.divide(k1,2)),stepSize),np.add(PulseIP,np.divide(k1,2)))
    k3 = np.multiply(op.BasicRKNL(gamma,np.add(PulseIP,np.divide(k2,2)),stepSize),np.add(PulseIP,np.divide(k2,2)))
    k4 = np.multiply(op.BasicRKNL(gamma,op.resolveBasicGVD(b2,stepSize,np.add(PulseIP,k3),samplingRate),stepSize),op.resolveBasicGVD(b2,stepSize,np.add(PulseIP,k3),samplingRate))
    #sum parts together
    s1 = np.add(PulseIP,np.divide(k1,6))
    s2 = np.add(s1,np.divide(k2,3))
    s3 = np.add(s2,np.divide(k3,3))
    #final step
    step = np.add(np.divide(k4,6),op.resolveBasicGVD(b2,stepSize,s3,samplingRate))
    return step