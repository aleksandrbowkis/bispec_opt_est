import numpy as np
import sys, os
from multiprocessing import Pool
from functools import partial
import time
sys.path.append('/home/amb257/software/cmplx_cmblensplus/wrap')
sys.path.append('/home/amb257/software/cmplx_cmblensplus/utils')
import curvedsky as cs
import basic

sys.path.append('/home/amb257/kappa_bispec/bispec_opt_est/N2_numerical/Binning_effects/full_n2_bias_calculation') # Add path to full_N2.py
sys.path.append('/home/amb257/kappa_bispec/bispec_opt_est/N2_numerical/Binning_effects/non_series') # Add path to full_non_series_N2.py
from full_N2 import do_N2_integral # New full N2 calculation
from Fold_no_series import usevegas_do_fold_no_series_integral # Import non-series N2 calculation 

sys.path.append('/home/amb257/kappa_bispec/bispec_opt_est/Configuration/') # Add path to configuration file include power spectra etc
from config import CMBConfig # Import CMBConfig class from config.py
# Import configuration class. can now do config.ctot_interp(l) etc.
config = CMBConfig()

#Function to compute the normalisation for a single triplet for binned bisepctrum estimator directly 
def N(L1, L2, L3):
    """Compute the normalisation factor for a single triplet L1, L2, L3 used in binned bispec estimator"""
    w3j = basic.wigner_funcs.wigner_3j(L3,L2,0,0)
    lower_bound_w3j = np.abs(L3 - L2)
    upper_bound_w3j = L3 + L2
    position_L1_in_w3j = L1 - lower_bound_w3j

    N = (2*L1+1)*(2*L2+1)*(2*L3+1) / (4*np.pi) * w3j[position_L1_in_w3j]**2
    
    return N

def find_triangles(bin_min, bin_max):
    """Vectorized triangle finding with pre-allocated arrays for better memory efficiency"""
    L = np.arange(bin_min, bin_max + 1)
    # Use numpy broadcasting instead of meshgrid for better memory usage
    L1 = L[:, None, None]
    L2 = L[None, :, None]
    L3 = L[None, None, :]
    
    mask = ((L1 + L2 > L3) & 
            (L2 + L3 > L1) & 
            (L3 + L1 > L2) & 
            (L3 <= L2) & 
            (L2 <= L1))
    
    # Find indices where mask is True
    valid_indices = np.where(mask)
    valid_triangles = np.column_stack([L[valid_indices[0]], 
                                     L[valid_indices[1]], 
                                     L[valid_indices[2]]])
    return valid_triangles

def find_folded_triangles(bin_min, bin_max):
    """Vectorized triangle finding for folded triangles"""
    L = np.arange(bin_min, bin_max + 1)
    L_half = np.arange(int(bin_min/2), int(bin_max/2) + 1)
    # Use numpy broadcasting instead of meshgrid for better memory usage
    L1 = L[:, None, None]
    L2 = L_half[None, :, None]
    L3 = L_half[None, None, :]
    #Note change mask so L3<L2<L1
    mask = ((L1 + L2 > L3) & 
            (L2 + L3 > L1) & 
            (L3 + L1 > L2) & 
            (L3 <= L2) & 
            (L2 <= L1))
    
    # Find indices where mask is True
    valid_indices = np.where(mask)
    valid_triangles = np.column_stack([L[valid_indices[0]], 
                                     L_half[valid_indices[1]], 
                                     L_half[valid_indices[2]]])
    return valid_triangles

def find_angles(L1, L2, L3):
    """Vectorized angle calculation"""
    theta1 = np.arccos((L2**2 + L3**2 - L1**2) / (2 * L2 * L3))
    theta2 = np.arccos((L1**2 + L3**2 - L2**2) / (2 * L1 * L3))
    theta3 = np.arccos((L1**2 + L2**2 - L3**2) / (2 * L1 * L2))
    
    x1 = np.zeros_like(L1)
    x2 = np.pi - theta3
    x3 = 2*np.pi - (theta1 + theta3)
    return x1, x2, x3

def process_triangle(triangle, config):
    """Process a single triangle - used by multiprocessing"""
    L1, L2, L3 = triangle
    x1, x2, x3 = find_angles(L1, L2, L3)
    return do_N2_integral(L1, L2, L3, x1, x2, x3, config.cl_phi_interp, config.ctot_interp, config.lcl_interp, config.ctotprime_interp, config.lclprime_interp, config.lcldoubleprime_interp, config.norm_factor_phi)

def process_triangle_noseries(triangle, config):
    """Process a single triangle using nonseries approximation - used by multiprocessing"""
    L1_mag, L2_mag, L3_mag = triangle
    x1, x2, x3 = find_angles(L1_mag, L2_mag, L3_mag)
    L1 = np.array([L1_mag*np.cos(x1), L1_mag*np.sin(x1)])
    L2 = np.array([L2_mag*np.cos(x2), L2_mag*np.sin(x2)])
    L3 = np.array([L3_mag*np.cos(x3), L3_mag*np.sin(x3)])

    return usevegas_do_fold_no_series_integral(L1, L2, L3, config)

def process_bin(bin_edges, config, normalising_factor, num_processes=None, fold=False, series=True):
    """Process all bins with multiprocessing"""
    bin_mid = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    averaged_N2_bin = []
    
    if fold == False:
        if series == True:
            # Create a pool of workers
            with Pool(processes=num_processes) as pool:
                for i in range(len(bin_edges) - 1):
                    bin_min = bin_edges[i]
                    bin_max = bin_edges[i + 1]
                    print(f"Processing bin {i+1}/{len(bin_edges)-1}: {bin_min}-{bin_max}")
                    
                    # Get triangles for this bin
                    triangles = find_triangles(bin_min, bin_max)
                    
                    # Process triangles in parallel. Partial creates specialised function with config as a fixed argument
                    N2_values = pool.map(partial(process_triangle, config=config), triangles)
                    
                    # Calculate average for this bin
                    if N2_values:
                        sum_N2 = np.sum(N2_values)
                    else:
                        sum_N2 = 0
                    bin_N2 = sum_N2 / normalising_factor[i]
                    averaged_N2_bin.append(bin_N2)
                    
                    # Print progress
                    print(f"Completed bin {i+1} with {len(triangles)} triangles")
        else:
            # Create a pool of workers
            with Pool(processes=num_processes) as pool:
                for i in range(len(bin_edges) - 1):
                    bin_min = bin_edges[i]
                    bin_max = bin_edges[i + 1]
                    print(f"Processing bin {i+1}/{len(bin_edges)-1}: {bin_min}-{bin_max}")
                    
                    # Get triangles for this bin
                    triangles = find_triangles(bin_min, bin_max)
                    
                    # Process triangles in parallel. Partial creates specialised function with config as a fixed argument
                    N2_values = pool.map(partial(process_triangle_noseries, config=config), triangles)
                    
                    # Calculate average for this bin
                    if N2_values:
                        sum_N2 = np.sum(N2_values)
                    else:
                        sum_N2 = 0
                    bin_N2 = sum_N2 / normalising_factor[i]
                    averaged_N2_bin.append(bin_N2)
                    
                    # Print progress
                    print(f"Completed NO SERIES bin {i+1} with {len(triangles)} triangles")
    else:
        if series == True:
            # Create a pool of workers
            with Pool(processes=num_processes) as pool:
                for i in range(len(bin_edges) - 1):
                    bin_min = bin_edges[i]
                    bin_max = bin_edges[i + 1]
                    print(f"Processing folded bin {i+1}/{len(bin_edges)-1}: {bin_min}-{bin_max}")

                    # Get triangles for this bin
                    triangles = find_folded_triangles(bin_min, bin_max)

                    # Process triangles in parallel
                    N2_values = pool.map(partial(process_triangle, config=config), triangles)

                    # Calculate average for this bin
                    if N2_values:
                        sum_N2 = np.sum(N2_values)
                    else:
                        sum_N2 = 0
                    bin_N2 = sum_N2 / normalising_factor[i]
                    averaged_N2_bin.append(bin_N2)
                    
                    # Print progress
                    print(f"Completed bin {i+1} with {len(triangles)} triangles")
        else:
            # Create a pool of workers
            with Pool(processes=num_processes) as pool:
                for i in range(len(bin_edges) - 1):
                    bin_min = bin_edges[i]
                    bin_max = bin_edges[i + 1]
                    print(f"Processing folded bin {i+1}/{len(bin_edges)-1}: {bin_min}-{bin_max}")

                    # Get triangles for this bin
                    triangles = find_folded_triangles(bin_min, bin_max)

                    # Process triangles in parallel
                    N2_values = pool.map(partial(process_triangle_noseries, config=config), triangles)

                    # Calculate average for this bin
                    if N2_values:
                        bin_N2 = np.sum(N2_values)
                    else:
                        bin_N2 = 0
                    bin_N2 = sum_N2 / normalising_factor[i]
                    averaged_N2_bin.append(bin_N2)
                    
                    # Print progress
                    print(f"Completed No SERIES bin {i+1} with {len(triangles)} triangles")
    
    return bin_mid, np.array(averaged_N2_bin)

def main():
    # Get number of CPUs and array task ID from SLURM
    #num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    #task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

    # Select bin edges for this task
    bin_edges = np.array([20,40,60])#,80,100,200])#np.array([20,40,60,80,100,200,300])
    
    # Time the execution
    start_time = time.time()
    
    # Series or no series?
    is_it_series = True
    is_it_folded = True
    normalising_factor=Nijk(bin_edges, len(bin_edges), fold=is_it_folded)
    print(normalising_factor)
    # Process bins for this task
    #bin_mid, averaged_N2_bin_equi = process_bin(bin_edges, config, num_processes=num_cpus, fold=False, series=is_it_series)
    # bin_mid, averaged_N2_bin_fold = process_bin(bin_edges, config, normalising_factor,
    #                                            num_processes=num_cpus, 
    #                                            fold=is_it_folded, 
    #                                            series=is_it_series)
    
    # Create filenames based on series flag and task ID
    series_str = 'series' if is_it_series else 'no_series'
    #output_eq_filename = f'../outputs/{series_str}_binned_equilateral_task_{task_id}.npy'
    output_fd_filename = f'../outputs/w3j_{series_str}_binned_folded_task_newfactor.npy'
    
    # Save results for this task
    #np.save(output_eq_filename, (bin_mid, averaged_N2_bin_equi))
    np.save(output_fd_filename, (bin_mid, averaged_N2_bin_fold))
    
    end_time = time.time()
    print(f"Task execution time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()