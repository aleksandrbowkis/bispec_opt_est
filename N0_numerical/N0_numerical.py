import numpy as np
import vegas 
from scipy.integrate import quad
from scipy.interpolate import interp1d
import sys, os
sys.path.append('/home/amb257/software/cmplx_cmblensplus/wrap')
sys.path.append('/home/amb257/software/cmplx_cmblensplus/utils')
# from cmblensplus/wrap/
import curvedsky as cs


# Parameters
lmax = 2000
Tcmb  = 2.726e6    # CMB temperature in microkelvin?
bstype = 'equi'
ellmin, ellmax = 2, 2000
rlmin, rlmax = 2, 2000 # CMB multipole range for reconstruction

# Define the directory where power spec are stored
input_dir = "../N2_numerical/power_spec"

# Load the power spec
L = np.load(os.path.join(input_dir, "L.npy"))
cl_phi = np.load(os.path.join(input_dir, "cl_phi.npy"))
lcl = np.load(os.path.join(input_dir, "lcl.npy"))
ctot = np.load(os.path.join(input_dir, "ctot.npy"))

# Now interpolate them
cl_phi_interp = interp1d(L, cl_phi, kind='cubic', bounds_error=False, fill_value="extrapolate")
lcl_interp = interp1d(L, lcl, kind='cubic', bounds_error=False, fill_value="extrapolate")
ctot_interp = interp1d(L, ctot, kind='cubic', bounds_error=False, fill_value="extrapolate")

################# Functions ##################

def dotprod(l1,l2):
    #Computes the dot product of two (multipole) vectors.
    l1dotl2 = l1[0]*l2[0] + l1[1]*l2[1]
    return l1dotl2

def vect_modulus(l1):
    #Computes the modulus of an input vector
    modl1 = np.sqrt(l1[0]**2 + l1[1]**2)
    return modl1

def response_func(CTlens, l1, l2, sizel1, sizel2):
    #Computes the response function f where <T(l1)T(l2)> = f(l1,l2)phi (T here is the lensed CMB temperature, phi is the lensing potential). Comes from taylor expanding the lensed temp.
    #CTlens is lensed power spectrum, l1, l2 are the multipole values.
    #L is the multipole at which we will evaluate the N(0) bias term
    L = l1 + l2
    f12 = dotprod(L,l1)*CTlens(sizel1) + dotprod(L,l2)*CTlens(sizel2)
    return f12

def bigF(l1, l2, l1size, l2size, CTlens, Ctotal):
    #computes the function F(l1,l2) = f(l1,l2)/(2 Ctot(l1) Ctot(l2))
    # f12 is the response function
    
    bigF = response_func(CTlens, l1, l2, l1size, l2size) / ( 2 * Ctotal(l1size) * Ctotal(l2size))
    return bigF

def make_equilateral_L(sizeL):
    # Makes a set of 3 L's satisfying the triangle condition w/ L1 along the x axis and each L same length.
    L1, L2, L3 = np.zeros(2), np.zeros(2), np.zeros(2)
    L1[0] = sizeL
    
    L2[0] = -sizeL * np.cos(np.pi/3)
    L2[1] = sizeL * np.sin(np.pi/3)

    L3[0] = -sizeL * np.cos(np.pi/3)
    L3[1] = -sizeL * np.sin(np.pi/3)
    
    return L1, L2, L3

def make_fold_L(sizeL):
    # Makes a set of 3 L's satisfying the condition L1 = 2L2 = 2L3 satisfying the triangle condition.
    L1, L2, L3 = np.zeros(2), np.zeros(2), np.zeros(2)
    L1[0] = sizeL

    L2[0] = -sizeL / 2
    L3[0] = -sizeL / 2

    return L1, L2, L3

################ Integrand function (low L no series expansion) ############

def integrand_N0(ell, L1, L3, lcl_interp, ctot_interp, ellmin, ellmax):
    
    sizeell = vect_modulus(ell)
    L1minusell = L1-ell
    sizeL1minusell = vect_modulus(L1minusell)
    ellplusL3 = ell + L3
    sizeellplusL3 = vect_modulus(ellplusL3)

    if ellmin <= sizeell <= ellmax and ellmin <= sizeL1minusell <= ellmax and ellmin <= sizeellplusL3 <= ellmax:
        F1 = bigF(ell, L1-ell, sizeell, sizeL1minusell, lcl_interp, ctot_interp)
        F13 = bigF(L1-ell, ell+L3, sizeL1minusell, sizeellplusL3, lcl_interp, ctot_interp)
        F3 = bigF(-ell, ell+L3, sizeell, sizeellplusL3, lcl_interp, ctot_interp)
        integrand = 1 / (2*np.pi)**2 * 8 * F1 * F13 * F3 * ctot_interp(sizeell) * ctot_interp(sizeL1minusell) * ctot_interp(sizeellplusL3)
    else:
        integrand = 0

    return integrand

################ Main code ####################

integration_limits = [[ellmin, ellmax], [ellmin, ellmax]] #2D integral over CMB multipoles ell
lensingLarray = np.arange(1,2000,5)
output = []

# Now calculate the normalisation. Outside loop as unnecessary to repeatedly calculate.
phi_norm, phi_curl_norm = {}, {}
phi_norm['TT'], phi_curl_norm['TT'] = cs.norm_quad.qtt('lens',lmax,rlmin,rlmax,lcl,ctot,lfac='')

for lensingL in lensingLarray:
    L1, L2, L3 = make_equilateral_L(lensingL)
    integrand_function = lambda ell: integrand_N0(ell, L1, L3, lcl_interp, ctot_interp, ellmin, ellmax) 
    integrator = vegas.Integrator(integration_limits)
    result = integrator(integrand_function, nitn=5, neval=1000)

    # Now output results
    from gvar import mean
    result_mean = mean(result)

    # Now calculate the normalisation for this lensingL. then normalise result. Recall we're using phi everywhere as the low L N2 result was computed for phi.
    # Also note we will want to plot N2 for kappa though for comparison.
    norm_factor_phi = phi_norm['TT'][int(lensingL)]
    result_mean *= norm_factor_phi**3 # Equilateral so all L equivalent.

    output.append(result_mean)

output = np.array(output) # Convert to numpy array
# Save results
# Create directory
output_dir = "N0_numerical_vegas"
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, "L_N0_num.npy"), lensingLarray)
np.save(os.path.join(output_dir, "N0_numerical_equi.npy"), output)