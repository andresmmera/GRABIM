import numpy as np
import GRABIM as GRABIM
from scipy.optimize import differential_evolution

test_status = np.array([])
verbose = 0

def Objective_Function(x, *args):
    ZS = args[0]
    ZL = args[1]
    freq = args[2] # Frequency vector
    code = args[3] # Ladder structure
    
    rho_k = GRABIM.LadderCircuitAnalysis.get_Input_Reflection_Coeff(ZS, ZL, code, x, freq);
    max_rho_k = 20*np.log10(max(np.abs(rho_k)));# Get the maximum and convert into dB scale
    
    
    return max_rho_k


################################  TEST 1 ########################################
if verbose:
    print("####################################################################")
    print("TEST 1: Objective function - Comparison vs Keysight ADS results")
    print("Description: Circuit analysis involving LS, CP, CS, LP and complex terminations")
    print("ZS = 50-j10 Ohm")
    print("LS = 9.78 nH")
    print("CP = 2.56 pF")
    print("CS = 6.37 pF")
    print("LP = 54.53 nH")
    print("ZL = 100+j50 Ohm")
    print("f1 = 5 MHz, f2 = 2000 MHz, 10 samples")


freq = np.array([5, 226.7, 448.3, 670, 891.7, 1113, 1335, 1557, 1778, 2000])*1e6;

ZS = (50-10j)*np.ones(len(freq), dtype=complex); # [Ohm] Source impedance
ZL = (100+50j)*np.ones(len(freq), dtype=complex); # [Ohm] Load impedance
code = ['LS', 'CP', 'CS', 'LP']
x = [9.78e-9, 2.56e-12, 6.37e-12, 54.53e-9]
T = GRABIM.LadderCircuitAnalysis.get_ABCD_Network(ZS, ZL, code, x, freq)

# Check S21
Result_Test_1a = GRABIM.LadderCircuitAnalysis.getS21_dB(T, ZS, ZL)
S21_Keysight_ADS = [-67.339, -3.31, -0.044, -0.092, -0.02, -0.127, -0.879, -2.427, -4.465, -6.621];
# Check S11
Result_Test_1b = GRABIM.LadderCircuitAnalysis.getS11_dB(T, ZS, ZL)
S11_Keysight_ADS = [-8.015e-7, -2.73, -19.946, -16.788, -23.313, -15.409, -7.372, -3.685, -1.923, -1.066];

test_a = 1;
test_b = 1;
if any(abs(Result_Test_1a - S21_Keysight_ADS) > 0.1):
    if verbose:
        print("FAILED S21")
    test_a = -1;
if any(abs(Result_Test_1b - S11_Keysight_ADS) > 0.1):
    if verbose:
        print("FAILED S11")
    test_b = -1;
    
if ((test_a == -1) or (test_b == -1)):
    print("TEST 1 FAILED")
    np.append(test_status, -1)
else:
    print("TEST 1 PASSED")
    np.append(test_status, 1)

################################  TEST 2a ########################################
if verbose:
    print("####################################################################")
    print("TEST 2a: Grid search")
    print("Description: Simple matching circuit RS, RL")
    print("ZS = 50 Ohm")
    print("Candidate network: [LS, CP]")
    print("ZL = 100 Ohm")
    print("f1 = 450 MHz, f2 = 500 MHz, 10 samples")

f1 = 450e6; # [Hz] Start frequency
f2 = 500e6; # [Hz] End frequency
m = 10; # Number of frequency samples
freq = np.linspace(f1, f2, m);

ZS = 50*np.ones(m, dtype=complex); # [Ohm] Source impedance
ZL = 100*np.ones(m, dtype=complex); # [Ohm] Load impedance

code = ['LS', 'CP'];
[v_best, RLmax] = GRABIM.GridSearch(ZS, ZL, freq, m, code)
expected_result = np.array([16.8e-9, 3.4e-12])

if ((abs(v_best[0]-expected_result[0]) > 0.1e-9) or (abs(v_best[1]-expected_result[1]) > 0.1e-12)):
    print("TEST 2a FAILED")
    np.append(test_status, -1)
else:
    print("TEST 2a PASSED")
    np.append(test_status, 1)
    
   
    
################################  TEST 2b ########################################
if verbose:
    print("####################################################################")
    print("TEST 2b: Grid search with ZS, ZL")
    print("Description: Simple matching circuit")
    print("ZS = 20-j10 Ohm")
    print("Candidate network: [LS, CP]")
    print("ZL = 120+j30 Ohm")
    print("f1 = 400 MHz, f2 = 1000 MHz, 10 samples")

f1 = 400e6; # [Hz] Start frequency
f2 = 1000e6; # [Hz] End frequency
m = 10; # Number of frequency samples
freq = np.linspace(f1, f2, m);

ZS = (20-10j)*np.ones(m, dtype=complex); # [Ohm] Source impedance
ZL = (120+30j)*np.ones(m, dtype=complex); # [Ohm] Load impedance

code = ['CS', 'LP', 'CS', 'LP', 'CS', 'LP'];
[v_best, RLmax] = GRABIM.GridSearch(ZS, ZL, freq, m, code)


if (RLmax < -10):
    print("TEST 2b FAILED")
    np.append(test_status, -1)
else:
    print("TEST 2b PASSED")
    np.append(test_status, 1)
    
    
    
################################  TEST 2c ########################################
if verbose:
    print("####################################################################")
    print("TEST 2c: Grid search with ZS, ZL vs freq")
    print("Description: Simple matching circuit")
    print("ZS = 20-j10 Ohm")
    print("Candidate network: [LS, CP]")
    print("ZL = 120+j30 Ohm")
    print("f1 = 400 MHz, f2 = 1000 MHz, 10 samples")

# Problem input
f1 = 400e6; # [Hz] Start frequency
f2 = 1000e6; # [Hz] End frequency
m = 10; # Number of frequency samples
freq = np.linspace(f1, f2, m);

ZS = [20-10j, 20-20j, 20-10j, 25-5j, 30+10j, 25-10j, 30-10j, 20-10j, 20-30j, 30-20j]; # [Ohm] Source impedance
ZL = [120+30j, 120+30j, 115+20j, 100+30j, 120+50j, 100, 120+30j, 120+30j, 100+30j, 100+20j, 110+30j]; # [Ohm] Load impedance

code = ['CS', 'LP', 'CS', 'LP', 'CS', 'LP'];
[v_best, RLmax] = GRABIM.GridSearch(ZS, ZL, freq, m, code)


if (RLmax < -10):
    print("TEST 2c FAILED")
    np.append(test_status, -1)
else:
    print("TEST 2c PASSED")
    np.append(test_status, 1)    
    
      
################################  TEST 6a ########################################
if verbose:
    print("####################################################################")
    print("TEST 6a: Grid search + Lagrane - Test from paper")
    print("Description: Broadband matching")
    print("Candidate network: ['LP', 'LS', 'CP', 'CS']")
    print("f1 = 0.3 rad/s, f2 = 1 rad/s, 10 samples")
    
# Problem input
f1 = 0.3/(2*np.pi); # [Hz] Start frequency
f2 = 1/(2*np.pi); # [Hz] End frequency
m = 10; # Number of frequency samples
freq = np.array([47.75, 60.13, 72.5, 84.88, 97.26, 109.6, 122, 134.4, 146.8, 159.2])*1e-3;

ZS = [0.713-0.452j, 0.594-0.491j, 0.480-0.500j, 0.377-0.485j, 0.287-0.452j, 0.212-0.409j, 0.151-0.358j, 0.103-0.304j, 0.066-0.248j, 0.038-0.192j]; # [Ohm] Source impedance
ZL = [0.590+0.717j, 0.695+0.744j, 0.769+0.763j, 0.820+0.784j, 0.857+0.809j, 0.884+0.837j, 0.904+0.870j, 0.919+0.906j, 0.932+0.944j, 0.941+0.985j]; # [Ohm] Load impedance


# GRID SEARCH
code = ['LP', 'LS', 'CP', 'CS'];
[v_best, RL] = GRABIM.GridSearch(ZS, ZL, freq, m, code, 1, delta_X = 1)
x_best = GRABIM.LagrangeConstrainedOptimizer(ZS, ZL, code, v_best, freq, Stop_Condition = 0.1, delta=4, scale=5)
RLmax = GRABIM.LadderCircuitAnalysis.get_Input_Reflection_Coeff(ZS, ZL, code, x_best, freq)
RLmax = np.max(20*np.log10(np.abs(RLmax)));
print("RLmax = ", RLmax)

if (RLmax > -5):
    print("TEST 6a FAILED")
    np.append(test_status, -1)
else:
    print("TEST 6a PASSED")
    np.append(test_status, 1)  

    
################################  TEST 6b ########################################
if verbose:
    print("####################################################################")
    print("TEST 6b: Differential Evolution - Test from paper")
    print("Description: Broadband matching")
    print("Candidate network: ['LP', 'LS', 'CP', 'CS']")
    print("f1 = 0.3 rad/s, f2 = 1 rad/s, 10 samples")  
    
bounds = [];
for i in range(0, len(code)):
    if ((code[i] == 'LS') or (code[i] == 'LP')):
        bounds.append((1e-12, 100))# Bounds for inductance
    elif ((code[i] == 'CS') or (code[i] == 'CP')):
        bounds.append((1e-15, 100))# Bounds for capacitance
bounds = tuple(bounds);

params = (ZS,  ZL, freq, code)

result = differential_evolution(Objective_Function, bounds, args=params, seed=1)  
RLmax = GRABIM.LadderCircuitAnalysis.get_Input_Reflection_Coeff(ZS, ZL, code, result['x'], freq)
RLmax = np.max(20*np.log10(np.abs(RLmax)));
print("RLmax = ", RLmax)

if (RLmax > -5):
    print("TEST 6b FAILED")
    np.append(test_status, -1)
else:
    print("TEST 6b PASSED")
    np.append(test_status, 1)    
    
    

if verbose: 
    print("##################### STATUS ################################")    
if (any(test_status==-1)):
    print("TESTBENCH FAILED")
else:
    print("TESTBENCH PASSED")