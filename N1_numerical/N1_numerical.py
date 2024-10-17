""" code to calculate the N1 bias to the reconstructed lensing bispectrum"""

import numpy as np
import vegas
from scipy.integrate import quad
from scipy.interpolate import interp1d
import sys, os
sys.path.append('/home/amb257/software/cmplx_cmblensplus/wrap')
sys.path.append('/home/amb257/software/cmplx_cmblensplus/utils')
# from cmblensplus/wrap/
import curvedsky as cs
import flatsky as fs
import multiprocessing as mp

# Parameters
lmax = 2000
Tcmb  = 2.726e6    # CMB temperature in microkelvin?
bstype = 'equi'
ellmin, ellmax = 2, 2000

# Define the directory where power spec are stored
input_dir = "../Power_spectra"

# Load the power spec
L = np.arange(0,2000+1,1)
ucl = np.loadtxt(os.path.join(input_dir, "unlensed_clTT_lmax8000.txt"))
gcl = np.loadtxt(os.path.join(input_dir, "glensed_clTT_lmax8000.txt"))
lcl = np.loadtxt(os.path.join(input_dir, "lensed_clTT_lmax8000.txt"))
cphi = np.loadtxt(os.path.join(input_dir, "clpp_lmax8000.txt"))
cphi = cphi[0:2001]
ucl = ucl[0:2001]
gcl = gcl[0:2001]
lcl = lcl[0:2001]

#Make noise power spectra
theta_fwhm = 1.4 #In arcminutes
sigma_noise = 10 #in muK-arcmin
arcmin2radfactor = np.pi / 60.0 / 180.0
noise_cl = (sigma_noise*arcmin2radfactor/Tcmb)**2*np.exp(L*(L+1.)*(theta_fwhm*arcmin2radfactor)**2/np.log(2.)/8.)
ctot = np.copy(lcl) + noise_cl

# Now interpolate them
#cl_phi_interp = interp1d(L, cl_phi, kind='cubic', bounds_error=False, fill_value="extrapolate")
ucl_interp = interp1d(L, ucl, kind='cubic', bounds_error=False, fill_value="extrapolate")
lcl_interp = interp1d(L, lcl, kind='cubic', bounds_error=False, fill_value="extrapolate")
gcl_interp = interp1d(L, gcl, kind='cubic', bounds_error=False, fill_value="extrapolate")
ctot_interp = interp1d(L, ctot, kind='cubic', bounds_error=False, fill_value="extrapolate")
cphi_interp = interp1d(L, cphi, kind='cubic', bounds_error=False, fill_value="extrapolate")

# Load the normalisation
flat_sky_norm = np.load("/home/amb257/kappa_bispec/bispec_opt_est/N0_numerical/normalisation/flat_sky_norm.npy")
print('shape', np.shape(flat_sky_norm))
flat_sky_norm_interp = interp1d(L, flat_sky_norm, kind='cubic', bounds_error=False, fill_value="extrapolate")

################# Functions ##################

def dotprod(l1, l2):
    # Computes the dot product of two sets of vectors. Note now vectorised so can compute dotproduct of sets of vectors rather than two single vectors.
    return np.sum(l1 * l2, axis=1)

def vect_modulus(l1):
    #Computes the modulus of an input vector
    modl1 = np.sqrt(np.sum(l1**2, axis=1))
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


@vegas.batchintegrand
def integrand_N1(x L1, L2, gcl_interp, ctot_interp, lcl_interp, cphi_interp, ellmin, ellmax):
    """
    Computes the integrand. Masks out regions
    """

    # x[:, 0] is the x-component of l1, x[:, 1] is the y-component of l2 then x[:, 2] is x component of l2 etc
    l1 = np.stack((x[:, 0], x[:, 1]))
    l2 = np.stack((x[:, 2], x[:, 3]))

    # Now compute the relevant vectors
    L1minusl1 = L1[np.newaxis, :] - l1
    L2minusl2 = L2[np.newaxis, :] - l2
    L1plusl2 = L1[np.newaxis, :] + l2
    l1plusl2 = l1 + l2

    sizel1 = np.sqrt(np.sum(l1**2, axis=1))
    sizel2 = np.sqrt(np.sum(l2**2, axis=1))
    sizeL1 = np.sqrt(np.sum(L1**2))
    sizeL1minusl1 = np.sqrt(np.sum(L1minusl1**2, axis=1))
    sizeL2minusl2 = np.sqrt(np.sum(L2minusl2**2, axis=1))
    sizeL1plusl2 = np.sqrt(np.sum(L1plusl2**2, axis=1))
    sizel1plusl2 = np.sqrt(np.sum(l1plusl2**2, axis=1))

    # mask out disallowed regions
    mask = (ellmin <= sizel1) & (sizel1 <= ellmax) & \
           (ellmin <= sizel2) & (sizel2 <= ellmax) & \
           (ellmin <= sizeL1minusl1) & (sizeL1minusl1 <= ellmax) & \
           (ellmin <= sizeL2minusl2) & (sizeL2minusl2 <= ellmax) & \
           (ellmin <= sizeL1plusl2) & (sizeL1plusl2 <= ellmax) & \
           (ellmin <= sizel1plusl2) & (sizel1plusl2 <= ellmax)
    
     # Initialize result array with zeros
    result = np.zeros_like(sizel1)
    valid_idx = np.where(mask)[0]

    # Weight functions
    F_l1_L1minusl1 = bigF(l1, L1minusl1, sizel1, sizeL1minusl1, gcl_interp, ctot_interp)
    F_l2_L2minusl2 = bigF(l2, L2minusl2, sizel2, sizeL2minusl2, gcl_interp, ctot_interp)
    F_L2minusl2_L1plusl2 = bigF(L2minusl2, L1plusl2, sizeL2minusl2, sizel1plusl2, gcl_interp, ctot_interp)

    # Response functions
    f_l1_L1minusl1 = response_func(lcl_interp, l1, L1minusl1, sizel1, sizeL1minusl1)
    f_l2_negL1plusl2 = response_func(lcl_interp, l2, -L1plusl2, sizel2, sizeL1plusl2)
    f_L1minusl1_negL1plusl2 = response_func(lcl_interp, L1minusl1, -L1plusl2, sizeL1minusl1, sizeL1plusl2)
    f_l1_l2 = response_func(lcl_interp, l1, l2, sizel1, sizel2)

    # Compute power spectra
    CTT_L2minusl2 = gcl_interp(sizeL2minusl2[valid_idx])
    Cphi_L1 = cphi_interp(sizeL1[valid_idx]) 
    Cphi_l1plusl2 = cphi_interp(sizel1plusl2[valid_idx])  

    # Combine all terms
    term1 = Cphi_L1[valid_idx] * f_l1_L1minusl1[valid_idx] * f_l2_negL1plusl2[valid_idx]
    term2 = 2 * Cphi_l1plusl2[valid_idx] * f_L1minusl1_negL1plusl2[valid_idx] * f_l1_l2[valid_idx]
    
    result = 4 * F_l1_L1minusl1[valid_idx] * F_l2_L2minusl2[valid_idx] * F_L2minusl2_L1plusl2[valid_idx] * CTT_L2minusl2[valid_idx] * (term1 + term2)
    
    return result


# Integrand function use vegas batchintegrand to evaluate integral at multiple points
@vegas.batchintegrand
def vegas_integrand(x, L1, L2, gcl_interp, ctot_interp, lcl_interp, ellmin, ellmax):
    # x[:, 0] is the x-component of l1, x[:, 1] is the y-component of l2 then x[:, 2] is x component of l2 etc
    l1 = np.column_stack((x[:, 0], x[:, 1]))
    l2 = np.column_stack((x[:, 2], x[:, 3]))

    # Now compute the relevant vectors
    L1minusl1 = L1[np.newaxis, :] - l1
    L2minusl2 = L2[np.newaxis, :] - l2
    L1plusl2 = L1[np.newaxis, :] + l2
    l1plusl2 = l1 + l2

    sizel1 = np.sqrt(np.sum(sizel1**2, axis=1))
    sizel2 = np.sqrt(np.sum(sizel2**2, axis=1))
    sizeL1minusl1 = np.sqrt(np.sum(L1minusl1**2, axis=1))
    sizeL2minusl2 = np.sqrt(np.sum(L2minusl2**2, axis=1))
    sizeL1plusl2 = np.sqrt(np.sum(L1plusl2**2, axis=1))
    sizel1plusl2 = np.sqrt(np.sum(l1plusl2**2, axis=1))

    # mask out disallowed regions
    mask = (ellmin <= sizel1) & (sizel1 <= ellmax) & \
           (ellmin <= sizel2) & (sizel2 <= ellmax) & \
           (ellmin <= sizeL1minusl1) & (sizeL1minusl1 <= ellmax) & \
           (ellmin <= sizeL2minusl2) & (sizeL2minusl2 <= ellmax) & \
           (ellmin <= sizeL1plusl2) & (sizeL1plusl2 <= ellmax) & \
           (ellmin <= sizel1plusl2) & (sizel1plusl2 <= ellmax)
    
    result = np.zeros(len(x))

    # Compute for valid indices
    valid_idx = np.where(mask)[0]
    if len(valid_idx) > 0:
        result[valid_idx] = integrand_N1(l1[valid_idx], l2[valid_idx], L1[valid_idx], L1minusl1[valid_idx], L2minusl2[valid_idx],L2minusl2[valid_idx], L1plusl2[valid_idx],gcl_interp, ctot_interp, lcl_interp)
    
    return result