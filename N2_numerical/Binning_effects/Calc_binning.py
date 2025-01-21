import numpy as np
import sys
from multiprocessing import Pool
from functools import partial
import time
sys.path.append('/home/amb257/software/cmplx_cmblensplus/wrap')
sys.path.append('/home/amb257/software/cmplx_cmblensplus/utils')
import curvedsky as cs
from full_N2_series_integral import calculate_N2

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
            (L1 <= L2) & 
            (L2 <= L3))
    
    # Find indices where mask is True
    valid_indices = np.where(mask)
    valid_triangles = np.column_stack([L[valid_indices[0]], 
                                     L[valid_indices[1]], 
                                     L[valid_indices[2]]])
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

def process_triangle(triangle):
    """Process a single triangle - used by multiprocessing"""
    L1, L2, L3 = triangle
    x1, x2, x3 = find_angles(L1, L2, L3)
    return calculate_N2(L1, L2, L3, x1, x2, x3)

def process_bin(bin_edges, num_processes=None):
    """Process all bins with multiprocessing"""
    bin_mid = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    averaged_N2_bin = []
    
    # Create a pool of workers
    with Pool(processes=num_processes) as pool:
        for i in range(len(bin_edges) - 1):
            bin_min = bin_edges[i]
            bin_max = bin_edges[i + 1]
            print(f"Processing bin {i+1}/{len(bin_edges)-1}: {bin_min}-{bin_max}")
            
            # Get triangles for this bin
            triangles = find_triangles(bin_min, bin_max)
            
            # Process triangles in parallel
            N2_values = pool.map(process_triangle, triangles)
            
            # Calculate average for this bin
            if N2_values:
                avg_N2 = np.mean(N2_values)
            else:
                avg_N2 = 0
            averaged_N2_bin.append(avg_N2)
            
            # Print progress
            print(f"Completed bin {i+1} with {len(triangles)} triangles")
    
    return bin_mid, np.array(averaged_N2_bin)

def main():
    # Define binning scheme
    bin_edges = np.array([2,20, 40, 60, 80, 100, 200, 300, 400, 500])
    
    # Time the execution
    start_time = time.time()
    
    # Process all bins
    bin_mid, averaged_N2_bin = process_bin(bin_edges)
    
    # Save results
    np.save('binning_tests_outputs/binned_equilateral.npy', (bin_mid, averaged_N2_bin))
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()