import numpy as np
import vegas 
from scipy.integrate import quad
from scipy.interpolate import interp1d
import sys, os
sys.path.append('/home/amb257/software/cmplx_cmblensplus/wrap')
sys.path.append('/home/amb257/software/cmplx_cmblensplus/utils')
# from cmblensplus/wrap/
import curvedsky as cs
from scipy.integrate import quad

# Parameters
lmax = 2000
Tcmb  = 2.726e6    # CMB temperature in microkelvin?
bstype = 'equi'
ellmin, ellmax = 2, 2000
rlmin, rlmax = 2, 2000 # CMB multipole range for reconstruction

# Define the directory where power spec are stored
input_dir = "power_spec"

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

def integrand_generator(L1, L2, L3, cl_phi_interp, lcl_interp, ocl_interp, ellmin, ellmax):
    """
    Closure for capturing fixed L1, L2, L3 etc. and returning the low L approximation of the N2 bias to the lensing bispectrum. Note no noise needed in numerator as the ps comes 
    from the <T^6> and gaussian noise will vanish for higher order correl fn.

    """

    def integrand_N2(ell):
        
        #print(ell.shape)
        sizeell = vect_modulus(ell)
        ellminusL1 = ell - L1
        sizeellminusL1 = vect_modulus(ellminusL1)
        ellminusL2 = ell - L2
        sizeellminusL2 = vect_modulus(ellminusL2)
        ellminusL3 = ell - L3
        sizeellminusL3 = vect_modulus(ellminusL3)
        ellplusL3 = ell + L3
        sizeellplusL3 = vect_modulus(ellplusL3)
        sizeL1 = vect_modulus(L1)
        sizeL2 = vect_modulus(L2)
        sizeL3 = vect_modulus(L3)

        if sizeell <= ellmax and sizeell >= ellmin and  sizeellminusL1 >= ellmin and sizeellminusL1 <= ellmax :
            Fint1int = bigF(ell, L1-ell, sizeell, sizeellminusL1, lcl_interp, ocl_interp)
            N2_A = 2/(2*np.pi)**4*Fint1int * cl_phi_interp(sizeL2) * cl_phi_interp(sizeL3) * lcl_interp(sizeell) * dotprod(ell, L2) * dotprod(ell, L3)   
        else:
            N2_A = 0

        if sizeell <= ellmax  and sizeell >= ellmin and  sizeellminusL1 >= ellmin and sizeellplusL3 <= ellmax and sizeellplusL3 >= ellmin and sizeellminusL1  <= ellmax:
            Fint1int = bigF(ell, L1-ell, sizeell, sizeellminusL1, lcl_interp, ocl_interp)
            N2_B = - 2/(2*np.pi)**4*Fint1int * cl_phi_interp(sizeL2) * cl_phi_interp(sizeL3) * lcl_interp(sizeellplusL3) * dotprod(ellplusL3, L2) * dotprod(ellplusL3, L3)   
        else:
            N2_B = 0

        return N2_A + N2_B
    return integrand_N2

################ Main #############

integration_limits = [[ellmin, ellmax], [ellmin, ellmax]] 
lensingLarray = np.arange(1,200,10)
output_vegas = []

# Now calculate the normalisation. Outside loop as unnecessary to repeatedly calculate.
phi_norm, phi_curl_norm = {}, {}
phi_norm['TT'], phi_curl_norm['TT'] = cs.norm_quad.qtt('lens',lmax,rlmin,rlmax,lcl,ctot,lfac='')

for lensingL in lensingLarray:
    L1, L2, L3 = make_equilateral_L(lensingL)
    integrand_function = integrand_generator(L1, L2, L3, cl_phi_interp, lcl_interp, ctot_interp, ellmin, ellmax) #second ps should be ctot_interp but changed to noiseless to cf giorgio's results 6/9/24
    integrator = vegas.Integrator(integration_limits)
    result = integrator(integrand_function, nitn=15, neval=1000)

    # Now output results
    from gvar import mean
    result_mean = mean(result)

    # Now calculate the normalisation for this lensingL. then normalise result. Recall we're using phi everywhere as the low L N2 result was computed for phi.
    # Also note we will want to plot N2 for kappa though for comparison.
    norm_factor_phi = phi_norm['TT'][int(lensingL)]
    result_mean *= norm_factor_phi

    output_vegas.append(result_mean)

# Change outputs to numpy arrays Factor of 3 for all equi. changed 1/(2pi)^4 as two from measure and two from term A and B def.
output_vegas = 3*np.array(output_vegas)

#Convert to kappa
#output_vegas *= (0.5*(lensingLarray*(lensingLarray+1)))^3

# Save results
# Create directory
output_dir = "vegas_results"
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, "L.npy"), lensingLarray)
np.save(os.path.join(output_dir, "VEGAS_lowL.npy"), output_vegas)

