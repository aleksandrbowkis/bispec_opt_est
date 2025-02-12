""" 
    Program to compute the low L approximation to the N2 bias term in reconstructed CMB lensing bispectrum.
"""

import numpy as np
from scipy.integrate import dblquad
from scipy.interpolate import interp1d
import sys, os
sys.path.append('/home/amb257/software/cmplx_cmblensplus/wrap')
sys.path.append('/home/amb257/software/cmplx_cmblensplus/utils')
# from cmblensplus/wrap/
import curvedsky as cs

################## Parameters ###############

lmax = 2000
Tcmb  = 2.726e6    # CMB temperature in microkelvin?
bstype = 'equi'
ellmin, ellmax = 2, 2000
rlmin, rlmax = 2, 2000

################## Load and interpolate power spectra ############

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

###################### Functions ################################

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

##################### Integrand ################################

def integrand_N2(ellx, elly, L1, L2, L3, lcl_interp, ctot_interp,cl_phi_interp):
    
    ell = np.array([ellx, elly])
    sizeell = vect_modulus(ell)
    ellminusL1 = ell - L1
    sizeellminusL1 = vect_modulus(ellminusL1)
    ellplusL3 = ell + L3
    sizeellplusL3 = vect_modulus(ellplusL3)
    sizeL2 = vect_modulus(L2)
    sizeL3 = vect_modulus(L3)

    if sizeell <= ellmax and sizeell >= ellmin and  sizeellminusL1 >= ellmin and sizeellminusL1 <= ellmax :
        Fint1int = bigF(ell, L1-ell, sizeell, sizeellminusL1, lcl_interp, ctot_interp)
        N2_A = 2/((2*np.pi)**2)*Fint1int * cl_phi_interp(sizeL2) * cl_phi_interp(sizeL3) * lcl_interp(sizeell) * dotprod(ell, L2) * dotprod(ell, L3)   
    else:
        N2_A = 0

    if sizeell <= ellmax  and sizeell >= ellmin and  sizeellminusL1 >= ellmin and sizeellplusL3 <= ellmax and sizeellplusL3 >= ellmin and sizeellminusL1  <= ellmax:
        Fint1int = bigF(ell, L1-ell, sizeell, sizeellminusL1, lcl_interp, ctot_interp)
        N2_B = - 2/((2*np.pi)**2)*Fint1int * cl_phi_interp(sizeL2) * cl_phi_interp(sizeL3) * lcl_interp(sizeellplusL3) * dotprod(ellplusL3, L2) * dotprod(ellplusL3, L3)   
    else:
        N2_B = 0

    return N2_A + N2_B

#################### Lets get integratingggggg woooo ###########

# Calculate the normalisation. Outside loop as unnecessary to repeatedly calculate.
phi_norm, phi_curl_norm = {}, {}
phi_norm['TT'], phi_curl_norm['TT'] = cs.norm_quad.qtt('lens',lmax,rlmin,rlmax,lcl,ctot,lfac='')

# What L's (lensing L) are we calculating the integral for? 
lensingLarray = np.arange(2,1000,20)
output = []

for lensingL in lensingLarray:
    L1, L2, L3 = make_equilateral_L(lensingL)
    result, error = dblquad(lambda x, y: integrand_N2(x, y, L1, L2, L3, lcl_interp, ctot_interp,cl_phi_interp), -ellmax, ellmax, lambda x: -ellmax, lambda x: ellmax, epsrel=1e-5)
    result *= phi_norm['TT'][int(lensingL)]

    output.append(result)

print(output)

output_dir = "scipy_results"
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, "L.npy"), lensingLarray)
np.save(os.path.join(output_dir, "2025scipy_lowL_noseries.npy"), output)