"""
    Computes the non series expansion but approximate N2 bias to the reconstructed lensing bispectrum. Can then be called for specific triangle configurations.
    Used in investigation of binning effect at low multipoles.
"""

import numpy as np
from scipy.integrate import dblquad
import sys
sys.path.append('/home/amb257/software/cmplx_cmblensplus/wrap')
sys.path.append('/home/amb257/software/cmplx_cmblensplus/utils')
sys.path.append('/home/amb257/kappa_bispec/bispec_opt_est/Configuration/')
import curvedsky as cs # from cmblensplus/wrap/
from config import CMBConfig # Import CMBConfig class from config.py

# Import config class
config = CMBConfig()

def non_series_N2_full_integrand(l, L1, L2, L3, xl, x1, x2, x3,config):
    """
    Calculate the integrand in approximate form of (but not series expansion) N2 bias to the reconstructed lensing bispectrum for any configuration
    Note this returns only one permutation the remaining two can be found by: L1<->L2 and L1<->L3
    
    Args:
        l, L1, L2, L3: Multipole values
        x1, x2, x3: Angle of each triangle side
        config: Configuration object with all necessary power spectra and interpolations
        
    Returns:
        float: Approximate integrand for N2 for general configuration for reconstructed lensing bispectrum for KAPPA!
    """
    # Get interpolated values
    Ctot = config.ctot_interp
    Ctt = config.lcl_interp
    Cpp = config.cl_phi_interp
    norm_phi = config.norm_factor_phi
    
    # Precompute some common cosine terms
    cos_x1_m_xl = np.cos(x1 - xl)
    cos_x2_m_xl = np.cos(x2 - xl)
    cos_x3_m_xl = np.cos(x3 - xl)
    cos_x2_m_x3 = np.cos(x2 - x3)

    # Precompute some common cosine arguments
    l_m_L1_square = l**2 + L1**2 - 2*l*L1*cos_x1_m_xl
    l_p_L3_square = l**2 + L3**2 + 2*l*L3*cos_x3_m_xl

    firstfactor = l*L1*L2*L3*norm_phi(L1)*Cpp(L2)*Cpp(L3)*(l*cos_x1_m_xl*Ctt(l) + (L1-l*cos_x1_m_xl)*Ctt(l_m_L1_square))
    secondfactor = l**2*cos_x2_m_xl*cos_x3_m_xl*Ctt(l) - (L3*cos_x2_m_x3 + l*cos_x2_m_xl) * (L3 + l*cos_x3_m_xl)*Ctt(l_p_L3_square)
    denominator = 4*(np.pi**2)*Ctot(l)*Ctot(l_m_L1_square)

    kappa_factor = L1*(L1+1) * L2*(L2+1) * L3*(L3+1) / 8

    integrand = kappa_factor*firstfactor*secondfactor/denominator
    return integrand

def do_non_series_integral(L1, L2, L3, x1, x2, x3, config, ellmin = 2, ellmax = 3000):
    """ Computes the integral defined in non_series_N2_full_integrand over xl: [0,2pi] and l: [ellmin, ellmax]
        This computes all the permutations (obtained by L1<->L2 and L1<->L3 in the definition of integrand)

        Args:
            L1, L2, L3: Multipole values
            x1, x2, x3: Angle of each triangle side
            config: Configuration object with all necessary power spectra and interpolations
        
        Returns:
            float: Integral value
    """

    # Using scipy's dblquad for numerical integration compute each permutation
    perm1, error = dblquad(lambda xl, l: non_series_N2_full_integrand(l, L1, L2, L3, xl, x1, x2, x3, config), 0, 2*np.pi, ellmin, ellmax)
    perm2, error = dblquad(lambda xl, l: non_series_N2_full_integrand(l, L2, L1, L3, xl, x2, x1, x3, config), 0, 2*np.pi, ellmin, ellmax)
    perm3, error = dblquad(lambda xl, l: non_series_N2_full_integrand(l, L3, L2, L1, xl, x3, x2, x1, config), 0, 2*np.pi, ellmin, ellmax)

    return perm1 + perm2 + perm3


if __name__ == '__main__':
    #This runs if program called as script
    # Define L values to compute integral for
    lensingLarray = np.arange(2, 1000, 10)
    # Output arrays
    output_fd = []
    output_eq = []

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

        integral_fd = do_non_series_integral(L1_fd, L2_fd, L3_fd, x1_fd, x2_fd, x3_fd, config, 2, 3000)
        integral_eq = do_non_series_integral(L1_eq, L2_eq, L3_eq, x1_eq, x2_eq, x3_eq, config, 2, 3000)

        output_fd.append(integral_fd)
        output_eq.append(integral_eq)

    # Change outputs to numpy arrays
    output_fd = np.array(output_fd)
    output_eq = np.array(output_eq)

    # Save result
    np.savetxt('../outputs/noseries_foldN2_from_full_int.txt', (lensingLarray, output_fd))
    np.savetxt('../outputs/noseries_equilN2_from_full_int.txt', (lensingLarray, output_eq))

