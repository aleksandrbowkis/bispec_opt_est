"""
    Computes the N2 bias to the reconstructed lensing bispectrum. Can then be called for specific triangle configurations.
    Used in investigation of binning effect at low multipoles
"""

import numpy as np
from numpy import random 
from scipy.integrate import quad
from scipy.interpolate import interp1d
import sys
sys.path.append('/home/amb257/software/cmplx_cmblensplus/wrap')
sys.path.append('/home/amb257/software/cmplx_cmblensplus/utils')
# from cmblensplus/wrap/
import curvedsky as cs

################ Parameters ###############

Tcmb  = 2.726e6    # CMB temperature in microkelvin?
rlmin, rlmax = 2, 3000 # CMB multipole range for reconstruction
nside = 2048
bstype = 'fold'
nsims = 448 # Number of simulations to average over (in sets of 3) 
ellmin = 2 
ellmax = 3000 ##### 30/09/24 CHANGED TO 3000 CF SIMULATED RESULTS

################ Power spectra ################

ls, cl_unl, cl_len, cl_phi = np.loadtxt('/home/amb257/kappa_bispec/make_sims_parallel/camb_lencl_phi.txt')
L = np.arange(rlmax+1)
Lfac = (L*(L+1.) / 2 )**2
lcl = cl_len[0:rlmax+1] / Tcmb**2
ucl = cl_unl[0:rlmax+1] / Tcmb**2 #dimless unlensed T Cl
cl_kappa = Lfac * cl_phi[0:rlmax+1]
cl_phi = cl_phi[0:rlmax+1]

#Make noise power spectra
theta_fwhm = 1.4 #In arcminutes
sigma_noise = 10 #in muK-arcmin
arcmin2radfactor = np.pi / 60.0 / 180.0
noise_cl = (sigma_noise*arcmin2radfactor/Tcmb)**2*np.exp(L*(L+1.)*(theta_fwhm*arcmin2radfactor)**2/np.log(2.)/8.)
ocl = np.copy(lcl) + noise_cl #INC NOISE CF SIMULATED RESULTS NB EQUI NOISELESS ATM TO CF GIORGIO

# Interpolation functions for cl_kappa and ucl
cl_phi_interp = interp1d(L, cl_phi, kind='cubic', bounds_error=False, fill_value="extrapolate")
lcl_interp = interp1d(L, lcl, kind='cubic', bounds_error=False, fill_value="extrapolate")
ctot_interp = interp1d(L, ocl, kind='cubic', bounds_error=False, fill_value="extrapolate")
ctotprime = np.gradient(ocl, L)
ctotprime_interp = interp1d(L, ctotprime, kind='cubic', bounds_error=False, fill_value="extrapolate")
lclprime = np.gradient(lcl, L)
lclprime_interp = interp1d(L, lclprime, kind='cubic', bounds_error=False, fill_value="extrapolate")
lcldoubleprime = np.gradient(lclprime, L)
lcldoubleprime_interp = interp1d(L, lcldoubleprime, kind='cubic', bounds_error=False, fill_value="extrapolate")

#### Integration functions ####

def calculate_n2_integrand(l, L1, L2, L3, x1, x2, x3, ctot_interp, lcl_interp, 
                     ctotprime_interp, lclprime_interp, lcldoubleprime_interp):
    """
    Calculate the N2 bias to the reconstructed lensing bispectrum for a specific
    configuration.
    
    Args:
        l, L1, L2, L3: Multipole values
        x1, x2, x3: Angular values
        ctot_interp: Interpolated total power spectrum (signal + noise)
        lcl_interp: Interpolated lensed CMB power spectrum
        ctotprime_interp: Interpolated derivative of total power spectrum
        lclprime_interp: Interpolated derivative of lensed power spectrum
        lcldoubleprime_interp: Interpolated second derivative of lensed power spectrum
    
    Returns:
        float: N2_integrand for folded configuration
    """
    # Get interpolated values
    Ctot = ctot_interp(l)
    Ctt = lcl_interp(l)
    Ctot_prime = ctotprime_interp(l)
    Ctt_prime = lclprime_interp(l)
    Ctt_doubleprime = lcldoubleprime_interp(l)
    
    # Precompute some common cosine terms
    cos_x1_m_x2 = np.cos(x1 - x2)
    cos_x1_p_x2_m_2x3 = np.cos(x1 + x2 - 2*x3)
    cos_x2_m_x3 = np.cos(x2 - x3)
    cos_3x1_m_x2_m_2x3 = np.cos(3*x1 - x2 - 2*x3)
    cos_2x1_p_x2_m_3x3 = np.cos(2*x1 + x2 - 3*x3)
    cos_2x1_m_x2_m_x3 = np.cos(2*x1 - x2 - x3)
    
    # First term in parentheses
    term1 = -2 * l * L1 * Ctot_prime * (
        8 * (3*cos_x1_m_x2 + cos_x1_p_x2_m_2x3) * Ctt**2 +
        2 * l * (13*cos_x1_m_x2 + 5*cos_x1_p_x2_m_2x3) * Ctt * Ctt_prime +
        l**2 * (6*cos_x1_m_x2 + cos_3x1_m_x2_m_2x3 + 
                3*cos_x1_p_x2_m_2x3) * Ctt_prime**2
    )
    
    # Second term in parentheses
    term2 = -Ctot * (
        64 * L3 * cos_x2_m_x3 * Ctt**2 -
        2 * l * Ctt * (
            (27*L1*cos_x1_m_x2 + 9*L1*cos_x1_p_x2_m_2x3 - 
             50*L3*cos_x2_m_x3) * Ctt_prime +
            3 * l * (3*L1*cos_x1_m_x2 + L1*cos_x1_p_x2_m_2x3 - 
                     2*L3*cos_x2_m_x3) * Ctt_doubleprime
        ) +
        l**2 * Ctt_prime * (
            (-18*L1*cos_x1_m_x2 + 3*L3*cos_2x1_p_x2_m_3x3 +
             L1*cos_3x1_m_x2_m_2x3 - 9*L1*cos_x1_p_x2_m_2x3 +
             13*L3*cos_2x1_m_x2_m_x3 + 34*L3*cos_x2_m_x3) * Ctt_prime +
            l * (-6*L1*cos_x1_m_x2 + L3*cos_2x1_p_x2_m_3x3 -
                 L1*cos_3x1_m_x2_m_2x3 - 3*L1*cos_x1_p_x2_m_2x3 +
                 3*L3*cos_2x1_m_x2_m_x3 + 6*L3*cos_x2_m_x3) * Ctt_doubleprime
        )
    )
    
    # Combine terms with prefactor
    prefactor = -(1 / (128 * np.pi * Ctot**3))
    
    result = prefactor * l * L1**2 * L2 * L3**2 * (term1 + term2)
    
    return result

######## Main function ########

##################### Test for fold ####################


# Define L values to compute integral for
lensingLarray = np.arange(2, 1000, 10)
# Output array
output_quad = []
# Normalisation
phi_norm, phi_curl_norm = {}, {}
phi_norm['TT'], phi_curl_norm['TT'] = cs.norm_quad.qtt('lens',rlmax,rlmin,rlmax,lcl,ocl,lfac='')
norm_factor_phi = interp1d(L, phi_norm['TT'], kind='cubic', bounds_error=False, fill_value="extrapolate")

#Define triangle shape for equilateral:
# x1 = 0
# x2 = 2*np.pi/3
# x3 = 4*np.pi/3


# Define triangle shape for folded:
x1 = 0
x2 = np.pi
x3 = np.pi

for lensingL in lensingLarray:
    #Define triangle side lengths (equilateral):
    # L1 = lensingL
    # L2 = lensingL
    # L3 = lensingL

    #Define triangle side lengths (folded):
    L1 = lensingL
    L2 = lensingL/2
    L3 = lensingL/2

    # Using scipy's quad for numerical integration
    
    result, error = quad(lambda l: calculate_n2_integrand(l, L1, L2, L3, x1, x2, x3, ctot_interp, lcl_interp, ctotprime_interp, lclprime_interp, lcldoubleprime_interp), ellmin, ellmax)

    output_quad.append(result)

# Change outputs to numpy arrays
output_quad = np.array(output_quad)

# Save result
np.savetxt('test_outputs/N2_from_full_int', (lensingLarray, output_quad))