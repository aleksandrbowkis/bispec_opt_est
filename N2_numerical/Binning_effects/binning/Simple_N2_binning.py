""" Code to check whether normalisation code for N2 outputs 1 if input unnormalised N2 is 1/"""
import numpy as np
import sympy as sp
import sys, os
from multiprocessing import Pool
from functools import partial
import time
sys.path.append('/home/amb257/software/cmplx_cmblensplus/wrap')
sys.path.append('/home/amb257/software/cmplx_cmblensplus/utils')
import curvedsky as cs
import basic

sys.path.append('/home/amb257/kappa_bispec/bispec_opt_est/Configuration/') # Add path to configuration file include power spectra etc
from config import CMBConfig # Import CMBConfig class from config.py
# Import configuration class. can now do config.ctot_interp(l) etc.
config = CMBConfig()

sys.path.append('/home/amb257/kappa_bispec/bispec_opt_est/N2_numerical/Binning_effects/full_n2_bias_calculation') # Add path to full_N2.py
from full_N2 import do_N2_integral # New full N2 calculation

# Define functions for normalisation, finding angles associates with triangle multipoles

def N(L1, L2, L3):
    """Compute the normalisation factor for arrays of triplets L1, L2, L3 used in binned bispec estimator"""
    w3j = basic.wigner_funcs.wigner_3j(L3,L2,0,0)
    lower_bound_w3j = np.abs(L3 - L2)
    position_L1_in_w3j = L1 - lower_bound_w3j
    if L1 >= lower_bound_w3j and np.abs(L1) <= np.abs(L3 + L2):
        N = (2*L1+1)*(2*L2+1)*(2*L3+1) * w3j[position_L1_in_w3j]**2 / (4*np.pi)
    else:
        #print("L1 out of bounds")
        N = 0
    return N

def N_bin(bin_edges, is_it_folded):
    """Compute the normalisation factor for all bins"""
    size_bin_edges = len(bin_edges)

    if is_it_folded == False:
        changebins = 1
    else:
        changebins = 2
    
    N = np.zeros(size_bin_edges-1)
    sum = 0
    for index, item in enumerate(bin_edges[0:size_bin_edges-1]):
        sum = 0
        lower_bound_bin = int(item)
        upper_bound_bin = int(bin_edges[index+1])
        for l3 in range(int(lower_bound_bin/changebins), int(upper_bound_bin/changebins)):
            for l2 in range(int(lower_bound_bin/changebins), int(upper_bound_bin/changebins)):
                #First calculate the l bounds of w3j function (allowed l1 values given l2,3)
                lower_bound_w3j = np.abs(l3 - l2)
                upper_bound_w3j = l3 + l2
                #Calculate the w3j's
                w3j = basic.wigner_funcs.wigner_3j(l3,l2,0,0)
                for l1 in range(lower_bound_bin, upper_bound_bin):
                    if l1 >= lower_bound_w3j and l1 <= upper_bound_w3j: 
                        position_l1_in_w3j = l1 - lower_bound_w3j #this is the position of the current value of l1 in the w3j array
                        sum += (2*l1+1)*(2*l2+1)*(2*l3+1) * w3j[position_l1_in_w3j]**2 / (4*np.pi)
        N[index] = sum
    return N

def find_angles(L1, L2, L3):
    """Vectorized angle calculation"""
    theta1 = np.arccos((L2**2 + L3**2 - L1**2) / (2 * L2 * L3))
    theta2 = np.arccos((L1**2 + L3**2 - L2**2) / (2 * L1 * L3))
    theta3 = np.arccos((L1**2 + L2**2 - L3**2) / (2 * L1 * L2))
    
    x1 = np.zeros_like(L1)
    x2 = np.pi - theta3
    x3 = 2*np.pi - (theta1 + theta3)
    return x1, x2, x3

# Define the worker function outside of bin_N2
def process_L1(args):
    """
    Process a single L1 value for a specific bin.
    
    Parameters:
    args (tuple): (bin_idx, L1, bin_edges, config)
    
    Returns:
    tuple: (bin_idx, L1_sum)
    """
    bin_idx, L1, bin_edges, config = args
    L2_array = np.arange(bin_edges[bin_idx], bin_edges[bin_idx+1])
    L3_array = np.arange(bin_edges[bin_idx], bin_edges[bin_idx+1])
    
    L1_sum = 0
    for L2 in L2_array:
        for L3 in L3_array:
            if N(L1, L2, L3) != 0:
                x1, x2, x3 = find_angles(L1, L2, L3)
                N2_unnorm = do_N2_integral(L1, L2, L3, x1, x2, x3, config.cl_phi_interp, 
                                        config.ctot_interp, config.lcl_interp, config.ctotprime_interp, 
                                        config.lclprime_interp, config.lcldoubleprime_interp, 
                                        config.norm_factor_phi)
                L1_sum += N(L1, L2, L3) * N2_unnorm
    
    return bin_idx, L1_sum

def process_L1_folded(args):
    """
    Process a single L1 value for a specific bin in folded case.
    
    Parameters:
    args (tuple): (bin_idx, L1, bin_edges, config)
    
    Returns:
    tuple: (bin_idx, L1_sum)
    """
    bin_idx, L1, bin_edges, config = args
    L2_array = np.arange(int(bin_edges[bin_idx]/2), int(bin_edges[bin_idx+1]/2))
    L3_array = np.arange(int(bin_edges[bin_idx]/2), int(bin_edges[bin_idx+1]/2))
    
    L1_sum = 0
    total_triangles = 0
    valid_triangles = 0
    for L2 in L2_array:
        for L3 in L3_array:
            total_triangles += 1
            if N(L1, L2, L3) != 0:
                valid_triangles += 1
                x1, x2, x3 = find_angles(L1, L2, L3)
                N2_unnorm = do_N2_integral(L1, L2, L3, x1, x2, x3, config.cl_phi_interp, 
                                        config.ctot_interp, config.lcl_interp, config.ctotprime_interp, 
                                        config.lclprime_interp, config.lcldoubleprime_interp, 
                                        config.norm_factor_phi)
                L1_sum += N(L1, L2, L3) * N2_unnorm                 
    #print(f"L1={L1}: Considered {total_triangles} triangles, {valid_triangles} were valid")
    return bin_idx, L1_sum


def bin_N2(bin_edges, config, fold=False, num_processes=None):
    """
    Calculate binned N2 bias term with parallelization over L1.
    
    Parameters:
    bin_edges (array): Edges for the multipole bins
    config (CMBConfig): Configuration object containing interpolation functions
    fold (bool): Whether to use folded binning (default: False)
    num_processes (int): Number of processes for parallelization (default: None, uses all available)
    
    Returns:
    array: N2 bias term for each bin
    """
    if fold == False:
        N_equi = N_bin(bin_edges, False) # Compute normalisation at bin level for equilateral case
        N2 = np.zeros(len(bin_edges) - 1)
        
        # Process each bin
        with Pool(processes=num_processes) as pool:
            for bin_idx in range(len(bin_edges) - 1):
                start_time = time.time()  # Track time for this bin
                
                # Get all L1 values in this bin
                L1_array = np.arange(bin_edges[bin_idx], bin_edges[bin_idx+1])
                
                # Create tasks for parallel processing - pass all necessary data
                tasks = [(bin_idx, L1, bin_edges, config) for L1 in L1_array]
                
                # Process all L1 values in parallel
                results = pool.map(process_L1, tasks)
                
                # Sum results and normalize by N_equi
                bin_sum = sum(result[1] for result in results)
                N2[bin_idx] = bin_sum / N_equi[bin_idx]
                
                # Report time taken for this bin
                print(f"Bin {bin_idx} ({bin_edges[bin_idx]}-{bin_edges[bin_idx+1]}) completed in {time.time() - start_time:.2f} seconds")
    else:
        N_fold = N_bin(bin_edges, True) # Compute normalisation at bin level for equilateral case
        N2 = np.zeros(len(bin_edges) - 1)
        
        # Process each bin
        with Pool(processes=num_processes) as pool:
            for bin_idx in range(len(bin_edges) - 1):
                start_time = time.time()  # Track time for this bin
                
                # Get all L1 values in this bin
                L1_array = np.arange(bin_edges[bin_idx], bin_edges[bin_idx+1])
                
                # Create tasks for parallel processing - pass all necessary data
                tasks = [(bin_idx, L1, bin_edges, config) for L1 in L1_array]
                
                # Process all L1 values in parallel
                results = pool.map(process_L1_folded, tasks)
                
                # Sum results and normalize by N_equi
                bin_sum = sum(result[1] for result in results)
                N2[bin_idx] = bin_sum / N_fold[bin_idx]
                
                # Report time taken for this bin
                print(f"Bin {bin_idx} ({bin_edges[bin_idx]}-{bin_edges[bin_idx+1]}) completed in {time.time() - start_time:.2f} seconds")
        
    return N2


# Update the main function to use parallelization
def main():

    is_it_folded = False
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ['true', 't', '1', 'yes', 'y']:
            is_it_folded = True
    
    bin_edges = np.array([20, 40, 60, 80, 100, 200,300, 400, 500,600, 700,800, 900, 1000])
    bin_mid = (bin_edges[1:] + bin_edges[:-1]) / 2
    # Use environment variables to determine process count
    import os
    # Either use all cores on the node, or a specific number per task
    total_cores = int(os.environ.get('SLURM_NTASKS', 1)) * int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    num_processes = max(1, total_cores - 2)  # Leave a couple cores for system tasks
    
    print(f"Using {num_processes} processes for parallelization")
    
    start_time = time.time()
    N2 = bin_N2(bin_edges, config, fold=is_it_folded, num_processes=num_processes)
    print('N2', N2)
    # Save results    
    fold_str = 'folded' if is_it_folded else 'equilateral'
    output_fd_filename = f'../outputs/Simple_N2_binned_{fold_str}.npy'
    np.save(output_fd_filename, (bin_mid, N2))

if __name__ == "__main__":
    main()
