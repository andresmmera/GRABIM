from scipy.optimize import minimize
from GRABIM import LadderCircuitAnalysis
import numpy as np

def Cost_Function(x, args):
    ZS = args[0]
    ZL = args[1]
    freq = args[2]
    code = args[3]
    x_np1 = x[-1];
    
    # Circuit analysis
    T = LadderCircuitAnalysis.get_ABCD_Network(ZS, ZL, code, x[:-1], freq);
    S21 = LadderCircuitAnalysis.getS21_nu(T, ZS, ZL);#S21
    P = 1/S21;
    
    # Cost function
    c = x_np1*np.ones(len(P)) - P# It is a vector with frequency points
    return c;

def Objective_Function_Lagrange(x, *args):
    ZS = args[0]
    ZL = args[1]
    freq = args[2] # Frequency vector
    code = args[3] # Ladder structure
    s = args[4]; # Weigths
    u = args[5]; # Offsets
    
    x_np1 = x[-1];
    
    C = Cost_Function(x, args);
    
    S = 0; # Cumulative sum
    for i in range(1, len(C)):
        S += s[i]*np.power(np.min(C[i] - u[i],0), 2);
    
    F = x_np1 + S;
    
    return F


def LagrangeConstrainedOptimizer(ZS, ZL, code, v_best, freq, Stop_Condition = 0.01, delta=2, scale=10):
    T = LadderCircuitAnalysis.get_ABCD_Network(ZS, ZL, code, v_best, freq);
    S21 = LadderCircuitAnalysis.getS21_nu(T, ZS, ZL);
    P = 1/np.abs(S21); # The insertion loss is complex in general

    print('P', P)

    # Table 6.4.1. Step 1
    # Initialize offset u=O, set si=l, and set added variable XN+1 to the
    # maximum error, ei = (Pi-gi), i=1 to m.

    s = np.ones(len(P));
    u = np.zeros(len(P));
    xnp1 = np.max(P)# Maximum insertion loss over frequency for the best point found by grid search
    x = np.append(v_best, xnp1);# xnp1 (maximum insertion loss) is the added variable

    print('Initial vector',  x)
    print('Circuit topology', code)

    params = (ZS,  ZL, freq, code, s, u);
    
    
    C = Cost_Function(x, params);
    C_abs_previous = np.abs(C); # Vector
    C_ninf = np.array(np.max(C_abs_previous));

    # Bounds
    bnds = [];
    for i in range(1, len(v_best)):
        bnds.append((0, None))# Bounds for the network elements
    bnds.append((0, None))# Bounds for the added variable
    bnds = tuple(bnds);


    iterations = 0
    while (True):
        iterations +=1
        # Table 6.4.1. Step 2
        params = (ZS,  ZL, freq, code, s, u)
        result = minimize(Objective_Function_Lagrange, x, args=params, method="nelder-mead")
        print("Status:", result['message'])
        print("F =", result['fun'])
        print("x_best = ", result['x'])

        x = result['x']; # Best point obtained
        C = Cost_Function(x, params);

        C_abs = np.abs(C); # Vector
        C_ninf = np.append(C_ninf, np.max(C_abs)); # Append last ||C||inf

        # Terminate if the cost function does not decrease
        if (abs(C_ninf[-1] - C_ninf[-2]) < 1e-2):
            break

        print('Cost function at iteration', iterations, ' = ', C_ninf[-1])
        # Step 3:
        # Stop if |c(x')|inf, is suitable small, but if ||c(x')||intfy increased, go to step 5.

        if (C_ninf[-1] < Stop_Condition):
            break
        elif(C_ninf[-1] < C_ninf[-2]):
            print('Step #4')
            # Step 4:
            #If each |c(x')| decreased by factor 4 or more, set u=u-c(x') and go
            #to step 2.
            #print("Did all C_abs decreased?")
            #print(C_abs < 0.25*C_abs_previous)
            if (all(C_abs < C_abs_previous/delta)):
                u = u-C;
                C_abs_previous = C_abs;
                continue;
        elif(C_ninf[-1] > C_ninf[-2]):
            # Step 5: 
            #  Corresponding to each \ci(x')| not decreasing by factor 4, adjust
            #  si=10Si and ui=ui/10, then go to step 2.

            for i in range(0, len(C_abs)):
                if (C_abs[i] > C_abs_previous[i]/delta):
                    s[i] = scale*s[i];
                    u[i] = u[i]/scale;
            print("New Lagrange parameters")
            print("s = ", s)
            print("u = ", u)
            C_abs_previous = C_abs;

    return result['x']