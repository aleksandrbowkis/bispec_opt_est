"""
    Computes the non series expansion but approximate N2 bias to the reconstructed lensing bispectrum in the FOLDED CONFIGURATION.
    Used to check validity of series expansion.
"""

import numpy as np
np.set_printoptions(precision=15)
from scipy.integrate import dblquad
import vegas
import multiprocessing as mp
from functools import partial
import os
import sys
sys.path.append('/home/amb257/software/cmplx_cmblensplus/wrap')
sys.path.append('/home/amb257/software/cmplx_cmblensplus/utils')
sys.path.append('/home/amb257/kappa_bispec/bispec_opt_est/Configuration/')
import curvedsky as cs # from cmblensplus/wrap/
from config import CMBConfig # Import CMBConfig class from config.py

# Import config class
config = CMBConfig()

def bigF(l: np.ndarray, L: np.ndarray, config) -> np.ndarray:
    """
    Compute lensing response function F(l1,l2) = f(l1,l2)/(2 Ctot(l1) Ctot(l2))
    with epsilon parameter to avoid division by zero.
    
    Parameters
    ----------
    l1, l2 : np.ndarray
        Multipole vectors, shape (..., 2)
    config : Config
        Configuration object containing power spectra
        
    Returns
    -------
    np.ndarray
        Response function 
    
    """
    l = np.asarray(l, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    
    # Compute sizes 
    l_size = np.sqrt(np.sum(l*l, axis=-1))
    L_m_l = L - l
    l_m_l_size = np.sqrt(np.sum(L_m_l*L_m_l, axis=-1))
    
    # Add small epsilon to avoid division by zero
    eps = 1e-10
    l_size = np.maximum(l_size, eps)
    l_m_l_size = np.maximum(l_m_l_size, eps)
    
    # Get power spectra
    Ctt = config.lcl_interp
    
    # Compute dot products explicitly. Ellipsis allows handling of vectors with any leading number of dimensions.
    dot_L_l = L[...,0]*l[...,0] + L[...,1]*l[...,1]
    dot_L_Lml = L[...,0]*L_m_l[...,0] + L[...,1]*L_m_l[...,1]
    
    # Response function with regularization
    f_lL = (dot_L_l * Ctt(l_size) + dot_L_Lml * Ctt(l_m_l_size))
    denominator = 2 * Ctt(l_size) * Ctt(l_m_l_size)
    denominator = np.maximum(denominator, eps)
    
    return f_lL / denominator

def fold_no_series_integrand(l, L1, L2, L3, config):
    """
    Calculate the integrand in approximate form of (but not series expansion) N2 bias to the reconstructed lensing bispectrum for FOLDED CONFIGURATION
    This is the sum of the type A and type B terms
    Note this returns only one permutation the remaining two can be found by: L1<->L2 and L1<->L3
    
    Args:
        l, L1, L2, L3: Multipole vectors
        config: Configuration object with all power spectra, normalisations and interpolations
        
    Returns:
        float: Approximate integrand for N2 for folded configuration for reconstructed lensing bispectrum for KAPPA!
    """
    # Ensure all inputs are float64
    l = np.asarray(l, dtype=np.float64)
    L1 = np.asarray(L1, dtype=np.float64)
    L2 = np.asarray(L2, dtype=np.float64)
    L3 = np.asarray(L3, dtype=np.float64)
    
    # Get interpolated values
    Ctot = config.ctot_interp
    Ctt = config.lcl_interp
    Cpp = config.cl_phi_interp
    norm_phi = config.norm_factor_phi
    
    # Compute sizes with improved stability
    eps = 1e-10
    l_size = np.sqrt(np.sum(l*l))
    L1_size = np.sqrt(np.sum(L1*L1))
    L2_size = np.sqrt(np.sum(L2*L2))
    L3_size = np.sqrt(np.sum(L3*L3))
    
    # Apply size thresholds
    l_size = np.maximum(l_size, eps)
    L1_size = np.maximum(L1_size, eps)
    L2_size = np.maximum(L2_size, eps)
    L3_size = np.maximum(L3_size, eps)
    
    # Compute l + L3 more carefully
    l_p_L3 = l + L3
    l_p_L3_size = np.sqrt(np.sum(l_p_L3*l_p_L3))
    l_p_L3_size = np.maximum(l_p_L3_size, eps)
    
    # Compute dot products 
    dot_L2_l = L2[0]*l[0] + L2[1]*l[1]
    dot_L3_l = L3[0]*l[0] + L3[1]*l[1]
    dot_L2_lpL3 = L2[0]*l_p_L3[0] + L2[1]*l_p_L3[1]
    dot_L3_lpL3 = L3[0]*l_p_L3[0] + L3[1]*l_p_L3[1]
    
    # Precompute common factors
    F_l_L1 = bigF(l, L1, config)
    kappa_factor = L1_size*(L1_size+1) * L2_size*(L2_size+1) * L3_size*(L3_size+1) / 8
    common_factor = norm_phi(L1_size) * F_l_L1 * Cpp(L2_size) * Cpp(L3_size) / (2*np.pi)**2
    
    # Compute terms with improved stability
    typeA = -2 * common_factor * Ctt(l_size) * dot_L2_l * dot_L3_l
    typeB = 2 * common_factor * Ctt(l_p_L3_size) * dot_L2_lpL3 * dot_L3_lpL3
    
    return kappa_factor * (typeA + typeB)

def do_fold_no_series_integral(L1, L2, L3, config, ellmin=2, ellmax=3000):
    """
    Computes the 2D integral over l for the folded N2 bias approximation.
    Integrates over magnitude |l| and angle θ, with the measure d²l/(2π)². Note the 1/(2π)² included in def of integrand.
    
    Args:
        L1, L2, L3: Multipole vectors (2D arrays)
        config: Configuration object with power spectra, QE normalisatoin
        ellmin, ellmax: Integration bounds for |l|
    
    Returns:
        float: Integral including all permutations
    """
    def integrand_2d(l_mag, theta):
        # Convert scalar inputs to double precision
        l_mag = np.float64(l_mag)
        theta = np.float64(theta)

        # Construct l from magnitude and angle
        l = np.array([l_mag * np.cos(theta), l_mag * np.sin(theta)])
        
        # Compute integral for each permutation
        perm1 = fold_no_series_integrand(l, L1, L2, L3, config)
        perm2 = fold_no_series_integrand(l, L2, L1, L3, config)
        perm3 = fold_no_series_integrand(l, L3, L2, L1, config)
        
        # Include the measure l_mag from using polar coordinates
        return (perm1 + perm2 + perm3) * l_mag
    
    result, error = dblquad(lambda theta, l_mag: integrand_2d(l_mag, theta), ellmin, ellmax, lambda x: 0, lambda x: 2*np.pi)
    
    return result



def usevegas_do_fold_no_series_integral(L1, L2, L3, config, ellmin=2, ellmax=3000):
    """
    Computes the 2D integral over l for the folded N2 bias approximation using Vegas integration.
    Integrates over magnitude |l| and angle θ, with the measure d²l/(2π)².
    
    Args:
        L1, L2, L3: Multipole vectors (2D arrays)
        config: Configuration object with power spectra, QE normalization
        ellmin, ellmax: Integration bounds for |l|
    
    Returns:
        float: Integral including all permutations
    """    
     @vegas.batchintegrand
    def integrand_2d(x):
        # Transform coordinates with importance sampling
        # Use log spacing for l_mag to better sample important regions
        log_l_mag = np.log(ellmin) + x[:, 0] * (np.log(ellmax) - np.log(ellmin))
        l_mag = np.exp(log_l_mag)
        theta = x[:, 1] * 2*np.pi
        
        # Compute l vectors
        l_x = l_mag * np.cos(theta)
        l_y = l_mag * np.sin(theta)
        l = np.column_stack([l_x, l_y])
        
        # Calculate results with vectorization where possible
        result = np.zeros(len(x))
        for i in range(len(x)):
            perm1 = fold_no_series_integrand(l[i], L1, L2, L3, config)
            perm2 = fold_no_series_integrand(l[i], L2, L1, L3, config)
            perm3 = fold_no_series_integrand(l[i], L3, L2, L1, config)
            # Include l_mag for polar coordinates and Jacobian for log transform int f dl = int f l d(logl)
            result[i] = (perm1 + perm2 + perm3) * l_mag[i] * l_mag[i]
        
        # Include remaining Jacobian factors
        return result * (np.log(ellmax) - np.log(ellmin)) * 2*np.pi

    # Initialize Vegas integrator with improved settings
    integ = vegas.Integrator([[0, 1], [0, 1]])
    
    # More careful warm-up phase
    warmup = integ(integrand_2d, nitn=15, neval=20000)
    
    # Final integration with increased iterations and evaluations
    result = integ(integrand_2d, nitn=100, neval=100000)
    
    if result.sdev / abs(result.mean) > 0.1:  # More than 10% relative error
        # Try again with more points
        result = integ(integrand_2d, nitn=150, neval=150000)
    
    return float(result.mean) # convert to float for multiprocessing

def compute_single_L(lensingL, config):
    """Compute integral for a single lensingL value with error checking"""
    L1 = np.array([lensingL, 0], dtype=np.float64)
    L2 = np.array([-lensingL/2, 0], dtype=np.float64)
    L3 = np.array([-lensingL/2, 0], dtype=np.float64)
    
    try:
        result = usevegas_do_fold_no_series_integral(L1, L2, L3, config)
        if not np.isfinite(result):
            print(f"Warning: Non-finite result for L={lensingL}")
            result = 0.0
    except Exception as e:
        print(f"Error computing L={lensingL}: {str(e)}")
        result = 0.0
        
    return lensingL, result

### Main function ###

if __name__ == '__main__':
    # Define L values
    lensingLarray = np.arange(2, 500, 10,  dtype=np.float64)
    
    # Create partial function where the config arg is fixed as pool can only take functions of one variable
    compute_func = partial(compute_single_L, config=config)

    # Instead of mp.cpu_count() - 1, use the SLURM allocated cores:
    num_processes = int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count() - 1))
    with mp.Pool(num_processes) as pool:
        results = pool.map(compute_func, lensingLarray)
    
    # Unzip results into separate arrays. Previously pool saves results as a list of tuples.
    lensingL_out, integrals = zip(*results)
    
    # Convert to numpy arrays
    lensingL_out = np.array(lensingL_out)
    integrals = np.array(integrals)
    
    # Save result
    np.savetxt('../outputs/logspace_vegas_fold_N2_no_series.txt', (lensingL_out, integrals))