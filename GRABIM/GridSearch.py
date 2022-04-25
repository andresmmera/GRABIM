import itertools
import numpy as np
from GRABIM import LadderCircuitAnalysis

# REFERENCES
# [1] Thomas R. Cuthbert. Broadband Direct-Coupled and Matching RF Networks. TRCPEP, 1999.
# [2] Wideband Circuit Design. Herbert J Carlin

# Generating matrix for the grid search as specified in [1], Eq. 6.3.5
def generating_matrix(dim):
    lst = list(itertools.product([0, 1], repeat=dim))
    arr = np.array(lst)

    ncols = arr.shape[0]
    for i in range(0, ncols):
        s = arr[i, :];
        s = [-1 if x==0 else x for x in s]
        arr[i, :] = s;

    return arr;

# Scale the values of the matching network previously normalized to fmax and 1 Ohm
# Reference: [2] Eq. 6.3.3 
def scaleValues_wrt_R_f(x, code, R, f):
    
    for i in range(len(x)):
        if ((code[i] == 'CS') or (code[i] == 'CP')):
            x[i] = x[i]/(R*f)
        elif ((code[i] == 'LS') or (code[i] == 'LP')):
            x[i] = R*x[i]/(f)
            
    return x


# GRID SEARCH ALGORITHM
# Reference [1]: Table 5.5.3. The GRABIM Grid Search Algorithm Without Details.
# INPUTS:
# ZS: Source impedance (without normalization)
# ZL: Load impedance (withour normalization)
# freq: Range of frequencies where the optimization is desired
# m: Number of frequency points where optimization must be done
# code: Candidate topology
#
# OUTPUT:
# v_scaled: Values of the network components (scaled)
def GridSearch(ZS, ZL, freq, m, code, verbose=0, delta_X=10):
    ####### Impedance and frequency normalization #######
    # Normalize impedance to 1 Ohm
    max_ZS = np.max(np.abs(ZS))
    max_ZL = np.max(np.abs(ZL))
    max_Z =  np.max([max_ZS, max_ZL])

    ZS_norm = ZS/max_Z;
    ZL_norm = ZL/max_Z;

    # Normalize frequency to 1 rad/s
    max_f = freq[-1]
    f_norm = freq/(max_f*2*np.pi);
    #####################################################

    # Log file
    if (verbose):
        f = open("GridSearch.log", "w")
    
    dim = len(code);

    # Grid building
    base_point = np.ones(dim); # Base point
    C = generating_matrix(dim);
    X = delta_X*base_point*C; # Space for data search

    rho_max = 1; # Maximum reflection coefficient
    x_best = base_point; # Best point

    n_searches = X.shape[0]; # Number of combinations

    while delta_X >= 0.025:
        found_better_x = 0;
        for k in range(0, n_searches):
            sk = delta_X*C[k];
            xk = base_point + sk;

            vk = np.exp(xk); # Convert the grid vector into a search space vector

            # Calculate the reflection coefficient over the whole frequency span
            rho_k = LadderCircuitAnalysis.get_Input_Reflection_Coeff(ZS_norm, ZL_norm, code, vk, f_norm);
            max_rho_k = np.max(np.abs(rho_k));# Get the maximum

            if (verbose):
                print("Testing: x=(", xk, '), v=(', vk, ') -> rho = ', max_rho_k, file = f )


            if (max_rho_k < rho_max): # A better combination was found
                rho_max = max_rho_k;
                x_best = xk;
                if (verbose):
                    print("A better point was found x=(", xk, "), v=", vk, ') -> rho =', rho_max, file = f)
                found_better_x = 1; # Then, recenter the grid and examine the search space again (same refinement factor)

        base_point = x_best;
        if (found_better_x == 0):
            # After examining the whole search space, shrink the search space around the best point by 1/4
            if (verbose):
                print("Shrinking grid (delta_x = deltax/4)", file = f)
            delta_X *= 0.25;
        else:
            if (verbose):
                print("Centering grid around: x=(", x_best, "), v=(", np.exp(x_best), ')', file = f)

    # Get reflection coefficient and VSWR of the best point        
    RL = LadderCircuitAnalysis.get_ReturnLoss_from_ReflectionCoefficient(rho_max);
    VSWR = LadderCircuitAnalysis.get_VSWR_from_ReflectionCoefficient(rho_max);
    if (verbose):
        print("Best point found:", x_best, file = f)
        print("Best rho:", rho_max, " RL = ", RL, " VSWR = ", VSWR, file = f)


    # Transform the grid point into the search space point
    v_best = np.exp(x_best);
    # Scale the result according to the previous normalization
    v_scaled = scaleValues_wrt_R_f(v_best, code, max_Z, max_f*2*np.pi);
    
    rho_max = LadderCircuitAnalysis.get_Input_Reflection_Coeff(ZS, ZL, code, v_scaled, freq);
    rho_max = np.max(np.abs(rho_max))
    RL = LadderCircuitAnalysis.get_ReturnLoss_from_ReflectionCoefficient(rho_max);

    if (verbose):
        print("Result (scaled)", v_scaled, file = f)
        f.close()
    return v_scaled, RL


def RemoveIrrelevantComponents(code, v_best, freq, ZS, ZL):
    
    max_ZS = np.max(np.abs(ZS))
    max_ZL = np.max(np.abs(ZL))
    max_Z =  np.max([max_ZS, max_ZL])
    
    min_ZS = np.min(np.abs(ZS))
    min_ZL = np.min(np.abs(ZL))
    min_Z =  np.min([min_ZS, min_ZL])
    
    k = 0
    index_to_remove = [];
    for comp in code:
        if (comp == 'LS'):
            w = 2*np.pi*freq[-1]; # Impedance at the highest frequency
            X = w*v_best[k];
            print('X(LS) = ', X*min_Z)
            if (X < 0.5*min_Z):
                # Remove component
                index_to_remove = np.append(index_to_remove, k)
        elif (comp == 'LP'):
            w = 2*np.pi*freq[0]; # Impedance at the lowest frequency
            X = w*v_best[k];
            print('X(LP) = ', X*max_Z)
            if (X > 5*max_Z):
                # Remove component
                index_to_remove = np.append(index_to_remove, k)
        elif (comp == 'CS'):
            w = 2*np.pi*freq[0]; # Impedance at the lowest frequency
            X = 1/(w*v_best[k]);
            print('X(CS) = ', X*min_Z)
            if (X < 0.33*max_Z):
                # Remove component
                index_to_remove = np.append(index_to_remove, k)
        elif (comp == 'CP'):
            w = 2*np.pi*freq[-1]; # Impedance at the highest frequency
            X = 1/(w*v_best[k]);
            print('X(CP) = ', X*max_Z)
            if (X > 5*max_Z):
                # Remove component
                index_to_remove = np.append(index_to_remove, k)
        k += 1
    
    # Remove irrelevant components
    print('To remove: ', index_to_remove)
    code = np.delete(code, index_to_remove)
    v_best = np.delete(v_best, index_to_remove)
    
    return [code, v_best]