"""
    Computes the N2 bias to the reconstructed lensing bispectrum. Can then be called for specific triangle configurations.
    Used in investigation of binning effect at low multipoles
"""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d # Still needed for norm_phi_interp
import sys
sys.path.append('/home/amb257/software/cmplx_cmblensplus/wrap')
sys.path.append('/home/amb257/software/cmplx_cmblensplus/utils')
sys.path.append('/home/amb257/kappa_bispec/bispec_opt_est/Configuration/')
import curvedsky as cs # from cmblensplus/wrap/
from config import CMBConfig # Import CMBConfig class from config.py

# Import config class
config = CMBConfig()

#### Integration functions ####

def calculate_n2_integrand(l, L1, L2, L3, x1, x2, x3, cphi_interp, ctot_interp, lcl_interp, 
                     ctotprime_interp, lclprime_interp, lcldoubleprime_interp, norm_phi_interp):
    """
    Calculate the N2 bias to the reconstructed lensing bispectrum for any configuration
    Note this returns only one permutation the remaining two can be found by: L1<->L2 and L1<->L3
    
    Args:
        l, L1, L2, L3: Multipole values
        x1, x2, x3: Angle of each triangle side
        ctot_interp: Interpolated total power spectrum (signal + noise)
        lcl_interp: Interpolated lensed CMB power spectrum
        ctotprime_interp: Interpolated derivative of total power spectrum
        lclprime_interp: Interpolated derivative of lensed power spectrum
        lcldoubleprime_interp: Interpolated second derivative of lensed power spectrum
    
    Returns:
        float: N2 for general configuration for reconstructed lensing bispectrum for KAPPA!
    """
    # Get interpolated values
    Ctot = ctot_interp(l)
    Ctt = lcl_interp(l)
    Ctot_prime = ctotprime_interp(l)
    Ctt_prime = lclprime_interp(l)
    Ctt_doubleprime = lcldoubleprime_interp(l)

    # Define all normalisations
    A_L1 = norm_phi_interp(L1)
    A_L2 = norm_phi_interp(L2)
    A_L3 = norm_phi_interp(L3)
    
    # Precompute some common cosine terms
    cos_x1_m_x2 = np.cos(x1 - x2)
    cos_x1_p_x2_m_2x3 = np.cos(x1 + x2 - 2*x3)
    cos_x2_m_x3 = np.cos(x2 - x3)
    cos_3x1_m_x2_m_2x3 = np.cos(3*x1 - x2 - 2*x3)
    cos_2x1_p_x2_m_3x3 = np.cos(2*x1 + x2 - 3*x3)
    cos_2x1_m_x2_m_x3 = np.cos(2*x1 - x2 - x3)
    
    # First term 
    term1 = -2 * l * L1 * Ctot_prime * (
        8 * (3*cos_x1_m_x2 + cos_x1_p_x2_m_2x3) * Ctt**2 +
        2 * l * (13*cos_x1_m_x2 + 5*cos_x1_p_x2_m_2x3) * Ctt * Ctt_prime +
        l**2 * (6*cos_x1_m_x2 + cos_3x1_m_x2_m_2x3 + 
                3*cos_x1_p_x2_m_2x3) * Ctt_prime**2
    )
    
    # Second term 
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
    kappa_factor = L1*(L1+1) * L2*(L2+1) * L3*(L3+1) / 8
    
    result = kappa_factor * prefactor * l * L1**2 * L2 * L3**2 * A_L1 * cphi_interp(L2) * cphi_interp(L3) * (term1 + term2)
    
    return result

######## Main function where test this in folded and equilateral limit ########
# Define L values to compute integral for
lensingLarray = np.arange(2, 1000, 10)
# Output arrays
output_fd = []
output_eq = []
# Normalisation
phi_norm, phi_curl_norm = {}, {}
phi_norm['TT'], phi_curl_norm['TT'] = cs.norm_quad.qtt('lens',config.rlmax,config.rlmin,config.rlmax,config.lcl,config.ocl,lfac='')
norm_factor_phi = interp1d(config.L, phi_norm['TT'], kind='cubic', bounds_error=False, fill_value="extrapolate")

#Define triangle shape for equilateral:
x1_eq = 0
x2_eq = 2*np.pi/3
x3_eq = 4*np.pi/3


# Define triangle shape for folded:
x1_fd = 0
x2_fd = np.pi
x3_fd = np.pi

for lensingL in lensingLarray:
    #Define triangle side lengths (equilateral):
    L1_eq = lensingL
    L2_eq = lensingL
    L3_eq = lensingL

    #Define triangle side lengths (folded):
    L1_fd = lensingL
    L2_fd = lensingL/2
    L3_fd = lensingL/2

    # Using scipy's quad for numerical integration
    perm1_fd, error = quad(lambda l: calculate_n2_integrand(l, L1_fd, L2_fd, L3_fd, x1_fd, x2_fd, x3_fd, config.cl_phi_interp, config.ctot_interp, config.lcl_interp, config.ctotprime_interp, config.lclprime_interp, config.lcldoubleprime_interp, norm_factor_phi), config.ellmin, config.ellmax)
    perm2_fd, error = quad(lambda l: calculate_n2_integrand(l, L2_fd, L1_fd, L3_fd, x2_fd, x1_fd, x3_fd, config.cl_phi_interp, config.ctot_interp, config.lcl_interp, config.ctotprime_interp, config.lclprime_interp, config.lcldoubleprime_interp, norm_factor_phi), config.ellmin, config.ellmax)
    perm3_fd, error = quad(lambda l: calculate_n2_integrand(l, L3_fd, L2_fd, L1_fd, x3_fd, x2_fd, x1_fd, config.cl_phi_interp, config.ctot_interp, config.lcl_interp, config.ctotprime_interp, config.lclprime_interp, config.lcldoubleprime_interp, norm_factor_phi), config.ellmin, config.ellmax)

    perm1_eq, error = quad(lambda l: calculate_n2_integrand(l, L1_eq, L2_eq, L3_eq, x1_eq, x2_eq, x3_eq, config.cl_phi_interp, config.ctot_interp, config.lcl_interp, config.ctotprime_interp, config.lclprime_interp, config.lcldoubleprime_interp, norm_factor_phi), config.ellmin, config.ellmax)
    perm2_eq, error = quad(lambda l: calculate_n2_integrand(l, L2_eq, L1_eq, L3_eq, x2_eq, x1_eq, x3_eq, config.cl_phi_interp, config.ctot_interp, config.lcl_interp, config.ctotprime_interp, config.lclprime_interp, config.lcldoubleprime_interp, norm_factor_phi), config.ellmin, config.ellmax)
    perm3_eq, error = quad(lambda l: calculate_n2_integrand(l, L3_eq, L2_eq, L1_eq, x3_eq, x2_eq, x1_eq, config.cl_phi_interp, config.ctot_interp, config.lcl_interp, config.ctotprime_interp, config.lclprime_interp, config.lcldoubleprime_interp, norm_factor_phi), config.ellmin, config.ellmax)
    
    integral_fd = perm1_fd + perm2_fd + perm3_fd
    integral_eq = perm1_eq + perm2_eq + perm3_eq
    output_fd.append(integral_fd)
    output_eq.append(integral_eq)

# Change outputs to numpy arrays
output_fd = np.array(output_fd)
output_eq = np.array(output_eq)

# Save result
np.savetxt('test_outputs/foldN2_from_full_int.txt', (lensingLarray, output_fd))
np.savetxt('test_outputs/equilN2_from_full_int.txt', (lensingLarray, output_eq))
