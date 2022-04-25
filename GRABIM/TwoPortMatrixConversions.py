import numpy as np
# IEEE Transactions on Microwave Theory and Techniques. Vol 42, No 2. February 1994.
# Conversions Between S, Z, Y, h, ABCD, and T Parameters which are Valid for Complex Source and Load Impedances.
# ABCD to S parameter matrix conversion
def TtoS(M, Z1, Z2):
    # Get the number of frequency samples
    dimensions = M.shape;
    
    if (len(dimensions) == 3):
        n_freq = M.shape[0];
    else:
        n_freq = 1;
        Z1 = np.array([Z1]);
        Z2 = np.array([Z2]);
        
    S = np.empty((n_freq, 2, 2), dtype=complex); # Array of m (2x2)-matrices
    for i in range(0, n_freq):
       
        T = M[i];
 
        S11 = T[0,0]*Z2[i] + T[0,1] - T[1,0]*np.conj(Z1[i])*Z2[i] - T[1,1]*np.conj(Z1[i]);
        S12 = np.linalg.det(T) * 2 * np.sqrt(np.real(Z1[i])*np.real(Z2[i]));
        S21 = 2*np.sqrt(np.real(Z1[i])*np.real(Z2[i]))
        S22 = -T[0,0]*np.conj(Z2[i]) + T[0,1] - T[1,0]*Z1[i]*np.conj(Z2[i]) + T[1,1]*np.conj(Z1[i]);
        
        den = T[0,0]*Z2[i] + T[0,1] + T[1,0]*Z1[i]*Z2[i] + T[1,1]*Z1[i]
        
        S[i] = np.array([[S11/den, S12/den], [S21/den, S22/den]]);
    
    return S

# S to ABCD parameter matrix conversion
def StoT(M, Z1, Z2):
    # Get the number of frequency samples
    dimensions = M.shape;
    if (len(dimensions) == 3):
        n_freq = M.shape[0];
    else:
        n_freq = 1;
        Z1 = np.array([Z1]);
        Z2 = np.array([Z2]);
        
    T = np.empty((n_freq, 2, 2), dtype=complex); # Array of m (2x2)-matrices
    for i in range(0, n_freq):
     
        S = M[i];

        A = (np.conj(Z1[i])+S[0,0]*Z1[i])*(1-S[1,1]) + S[1,0]*S[0,1]*Z1[i];
        B = (np.conj(Z1[i])+S[0,0]*Z1[i])*(np.conj(Z2[i])+S[1,1]*Z2[i])-S[1,0]*S[0,1]*Z1[i]*Z2[i]
        C = (1 - S[0,0])*(1 - S[1,1])-S[0,1]*S[1,0]
        D = (1 - S[0,0])*(np.conj(Z2[i]) + S[1,1]*Z2[i]) + S[0,1]*S[1,0]*Z2[i];
    
        den = 2*S[1,0]*np.sqrt(np.real(Z1[i]) * np.real(Z2[i]))
    
        T[i] = np.array([[A/den, B/den], [C/den, D/den]]);
    
    return T


