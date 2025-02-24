"""
    Computes the non series expansion but approximate N2 bias to the reconstructed lensing bispectrum in the FOLDED CONFIGURATION.
    Used to check validity of series expansion.
    Currently rewriting to use numba. Note numba cannot use config class and cannot handle interpolated functions. 
    TO DO: Write function to do linear interpolation of power spectra and normalisation factors.

"""
from numba import njit
import numpy as np
import vegas
np.set_printoptions(precision=15)
from scipy.integrate import dblquad
import multiprocessing as mp
from functools import partial
import os
import sys
import time
sys.path.append('/home/amb257/software/cmplx_cmblensplus/wrap')
sys.path.append('/home/amb257/software/cmplx_cmblensplus/utils')
sys.path.append('/home/amb257/kappa_bispec/bispec_opt_est/Configuration/')
import curvedsky as cs # from cmblensplus/wrap/
from config import CMBConfig # Import CMBConfig class from config.py

# Import config class. Must load all the interpolated stuff here as numba doens't work with config class
config = CMBConfig()

# Define interpolation functions for power spectra and normalisation factors
def make_numba_functions(config, ellmax=3000):
    """Create Numba-compatible interpolation functions"""
    # Pre-compute values for interpolation
    ell_values = np.arange(1, ellmax+1)
    Ctt_values = np.array([config.lcl_interp(ell) for ell in ell_values])
    Cpp_values = np.array([config.cl_phi_interp(ell) for ell in ell_values])
    norm_phi_values = np.array([config.norm_factor_phi(ell) for ell in ell_values])
    
    @njit
    def Ctt_func(ell):
        if ell < 1 or ell > ellmax:
            return 0.0
        # Simple linear interpolation
        idx_float = ell - 1
        idx_low = int(np.floor(idx_float))
        idx_high = min(int(np.ceil(idx_float)), ellmax-1)
        if idx_low == idx_high:
            return Ctt_values[idx_low]
        weight = idx_float - idx_low
        return Ctt_values[idx_low] * (1-weight) + Ctt_values[idx_high] * weight
    
    @njit
    def Cpp_func(ell):
        if ell < 1 or ell > ellmax:
            return 0.0
        # Simple linear interpolation
        idx_float = ell - 1
        idx_low = int(np.floor(idx_float))
        idx_high = min(int(np.ceil(idx_float)), ellmax-1)
        if idx_low == idx_high:
            return Cpp_values[idx_low]
        weight = idx_float - idx_low
        return Cpp_values[idx_low] * (1-weight) + Cpp_values[idx_high] * weight
    
    @njit
    def norm_phi_func(ell):
        if ell < 1 or ell > ellmax:
            return 0.0
        # Simple linear interpolation
        idx_float = ell - 1
        idx_low = int(np.floor(idx_float))
        idx_high = min(int(np.ceil(idx_float)), ellmax-1)
        if idx_low == idx_high:
            return norm_phi_values[idx_low]
        weight = idx_float - idx_low
        return norm_phi_values[idx_low] * (1-weight) + norm_phi_values[idx_high] * weight
    
    return Ctt_func, Cpp_func, norm_phi_func


@njit
def bigF(l: np.ndarray, L: np.ndarray, Ctt_func) -> np.ndarray:
    """
    Compute lensing response function F(l1,l2) = f(l1,l2)/(2 Ctot(l1) Ctot(l2))
    for any pair of input multipole vectors.
    
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

    # Convert inputs to double precision
    l = np.array(l, dtype=np.float64)
    L = np.array(L, dtype=np.float64)

    l_size = np.sqrt(l[0]**2 + l[1]**2)
    L_minus_l = L - l
    l_m_l_size = np.sqrt(L_minus_l[0]**2 + L_minus_l[1]**2)
    L_dot_l = L[0]*l[0] + L[1]*l[1]
    L_dot_Lml = L[0]*L_minus_l[0] + L[1]*L_minus_l[1]
    f_lL = L_dot_l * Ctt_func(l_size) + L_dot_Lml * Ctt_func(l_m_l_size)
    

    denominator = 2 * Ctt_func(l_size) * Ctt_func(l_m_l_size)

    # Final result F(l,L)
    return f_lL / denominator

@njit
def fold_no_series_integrand(l, L1, L2, L3, Ctt_func, Cpp_func, norm_phi_func, ellmin=2, ellmax=3000):
    """
    Calculate the integrand in approximate form of (but not series expansion) N2 bias to the reconstructed lensing bispectrum for FOLDED CONFIGURATION
    This is the sum of the type A and type B terms
    Note this returns only one permutation the remaining two can be found by: L1<->L2 and L1<->L3
    Returns zero if outside of the region lmin < |L1-l| < lmax
    
    Args:
        l, L1, L2, L3: Multipole vectors
        config: Configuration object with all power spectra, normalisations and interpolations
        
    Returns:
        float: Approximate integrand for N2 for folded configuration for reconstructed lensing bispectrum for KAPPA!
    """

    # Convert all inputs to double precision
    l = np.array(l, dtype=np.float64)
    L1 = np.array(L1, dtype=np.float64)
    L2 = np.array(L2, dtype=np.float64)
    L3 = np.array(L3, dtype=np.float64)
    
    l_m_L1_size = np.linalg.norm(L1-l)
    if l_m_L1_size < ellmin or l_m_L1_size > ellmax:
        return 0.0

    # Get size L1, L2, L3, l + L3
    l_size = np.sqrt(l[0]**2 + l[1]**2)
    L1_size = np.sqrt(L1[0]**2 + L1[1]**2)
    L2_size = np.sqrt(L2[0]**2 + L2[1]**2)
    L3_size = np.sqrt(L3[0]**2 + L3[1]**2)
    
    l_plus_L3 = l + L3
    l_p_L3_size = np.sqrt(l_plus_L3[0]**2 + l_plus_L3[1]**2)

    # Precompute the lensing response function
    F_l_L1 = bigF(l, L1, Ctt_func)

    # Dot products
    L2_dot_l = L2[0]*l[0] + L2[1]*l[1]
    L3_dot_l = L3[0]*l[0] + L3[1]*l[1]
    L2_dot_lpL3 = L2[0]*l_plus_L3[0] + L2[1]*l_plus_L3[1]
    L3_dot_lpL3 = L3[0]*l_plus_L3[0] + L3[1]*l_plus_L3[1]

    # Factor to convert phi to kappa
    kappa_factor = L1_size*(L1_size+1) * L2_size*(L2_size+1) * L3_size*(L3_size+1) / 8

    typeA = -2.0*norm_phi_func(L1_size)*F_l_L1*Cpp_func(L2_size)*Cpp_func(L3_size)*Ctt_func(l_size)*L2_dot_l*L3_dot_l * (1/(2*np.pi)**2)
    typeB = 2.0*norm_phi_func(L1_size)*F_l_L1*Cpp_func(L2_size)*Cpp_func(L3_size)*Ctt_func(l_p_L3_size)*L2_dot_lpL3*L3_dot_lpL3 * (1/(2*np.pi)**2)

    return kappa_factor*(typeA + typeB) 


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
    # Get Numba-compatible interpolation functions
    Ctt_func, Cpp_func, norm_phi_func = make_numba_functions(config, ellmax)
    
    @vegas.batchintegrand
    def integrand_2d(x):
        """
        Vegas integrand function that handles the coordinate transformation and computation.
        x is an array of points in [0,1]^2 that Vegas samples.
        """
        # Transform coordinates
        # Use log spacing for l_mag to better sample important regions
        # log_l_mag = np.log(ellmin) + x[:, 0] * (np.log(ellmax) - np.log(ellmin))
        # l_mag = np.exp(log_l_mag)
        # theta = x[:, 1] * 2*np.pi

        # # Transform from [0,1]^2 to actual integration domain
        l_mag = x[:, 0] * (ellmax - ellmin) + ellmin
        theta = x[:, 1] * 2*np.pi
        
        # Compute l vectors for each point
        l_x = l_mag * np.cos(theta)
        l_y = l_mag * np.sin(theta)
        l = np.column_stack([l_x, l_y])
        
        # Calculate result for each point
        result = np.zeros(len(x))
        for i in range(len(x)):
            # Compute integral for each permutation
            perm1 = fold_no_series_integrand(l[i], L1, L2, L3, Ctt_func, Cpp_func, norm_phi_func, ellmin, ellmax)
            perm2 = fold_no_series_integrand(l[i], L2, L1, L3, Ctt_func, Cpp_func, norm_phi_func, ellmin, ellmax)
            perm3 = fold_no_series_integrand(l[i], L3, L2, L1, Ctt_func, Cpp_func, norm_phi_func, ellmin, ellmax)
            
            # Include the Jacobian factor from coordinate transformation
            # l_mag comes from the polar coordinate transformation
            result[i] = (perm1 + perm2 + perm3) * l_mag[i] #* l_mag[i] #Add back last factorif log spacing used
        
        # Include remaining Jacobian factors
        #return result * (np.log(ellmax) - np.log(ellmin)) * 2*np.pi #log spacing result
        return result * (ellmax - ellmin) * 2*np.pi
    
    # Create Vegas integrator
    integ = vegas.Integrator([[0, 1], [0, 1]])
    
    # Do a warm-up integration to adapt the grid
    warmup = integ(integrand_2d, nitn=1, neval=500)
    
    # Perform the final integration
    result = integ(integrand_2d, nitn=4, neval=6000)
    
    return float(result.mean)  # Explicitly convert to float for multiprocessing

def compute_single_L(lensingL, config):
    """Compute integral for a single lensingL value. Needed for multiprocessing. Defines L1,2,3 in folded instance."""
    L1 = np.array([lensingL, 0])
    L2 = np.array([-lensingL/2, 0])
    L3 = np.array([-lensingL/2, 0])
    
    return lensingL, usevegas_do_fold_no_series_integral(L1, L2, L3, config)

def compute_single_L_equi(lensingL, config):
    """Compute integral for a single lensingL value. Needed for multiprocessing. Defines L1,2,3 in equilateral instance."""
    L1 = np.array([lensingL, 0])
    L2 = np.array([-lensingL*np.cos(np.pi/3), lensingL*np.sin(np.pi/3)])
    L3 = np.array([-lensingL*np.cos(np.pi/3), -lensingL*np.sin(np.pi/3)])
    
    return lensingL, usevegas_do_fold_no_series_integral(L1, L2, L3, config)

### Main function ###

if __name__ == '__main__':

    # Time the execution
    start_time = time.time()

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

    end_time = time.time()
    print(f"execution time: {end_time - start_time:.2f} seconds")
    
    # Save result
    np.savetxt('../outputs/numba_vegas_fold_N2_no_series.txt', (lensingL_out, integrals))