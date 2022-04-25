import numpy as np
from GRABIM import TwoPortMatrixConversions

# Get the ABCD matrix of a ladder element
def get_ABCD_Matrix_Ladder_Element(code, x, w):
    x = np.atleast_1d(x)
    if (code == 'LS'):
        T = np.array([[1, 1j*w*x[0]], [0, 1]]);
    elif (code == 'LP'):
        T = np.array([[1, 0], [-1j/(w*x[0]), 1]]);
    elif (code == 'CS'):
        T =  np.array([[1, -1j/(w*x[0])], [0, 1]]);
    elif (code == 'CP'):
        T =  np.array([[1, 0], [1j*w*x[0], 1]]);
    elif (code == 'CASTL'):# x[0]: Z0, x[1]: theta
        T =  np.array([[cos(x[1]), x[0]*sin(x[1])], [sin(x[1])/x[0], cos(x[1])]]);
    return T

# Calculates the ABCD parameters of the whole network
def get_ABCD_Network(ZS, ZL, code, x, freq):
    RS = np.real(ZS);
    XS = np.imag(ZS);
    RL = np.real(ZL);
    XL = np.imag(ZL);

    m = freq.size;

    T_vs_f = np.empty((m, 2, 2), dtype=complex); # Array of m (2x2)-matrices
    for i in range(0, m): # Frequency loop      
        k = 0;
        for block in code: # Get ABCD for one frequency
            Tk = get_ABCD_Matrix_Ladder_Element(block, x[k], 2*np.pi*freq[i]);

            if (k == 0):
                T = Tk;
            else:
                T = np.matmul(T, Tk);
                
            k += 1;  
        T_vs_f[i] = T; # Store ABCD matrix for the i-th frequency
        
    return T_vs_f

# Reflection coefficient
def get_Input_Reflection_Coeff(ZS, ZL, code, x, freq):
    T = get_ABCD_Network(ZS, ZL, code, x, freq); # ABCD matrix calculation
    S = TwoPortMatrixConversions.TtoS(T, ZS, ZL); # S-parameter matrix conversion
    
    return S[:, 0,0] # Input reflection coefficient

# Get S21 in dB
def getS21_dB(T, ZS, ZL):
    S = TwoPortMatrixConversions.TtoS(T, ZS, ZL);
    return 20*np.log10(abs(S[:, 1,0]));

# Get S21 in natural units
def getS21_nu(T, ZS, ZL):
    S = TwoPortMatrixConversions.TtoS(T, ZS, ZL);
    return abs(S[:, 1,0]);

# Get S11 in dB
def getS11_dB(T, ZS, ZL):
    S = TwoPortMatrixConversions.TtoS(T, ZS, ZL);
    return 20*np.log10(abs(S[:, 0,0]));

# Get S11 in natural units
def getS11_nu(T, ZS, ZL):
    S = TwoPortMatrixConversions.TtoS(T, ZS, ZL);
    return abs(S[:, 0,0]);

# Return Loss from reflection coefficient
def get_ReturnLoss_from_ReflectionCoefficient(gamma):
    RL = -20*np.log10(gamma)
    return RL

# VSWR
def get_VSWR_from_ReflectionCoefficient(gamma):
    VSWR = (1 + gamma)/(1 - gamma);
    return VSWR;