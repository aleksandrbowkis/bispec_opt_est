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
        print("L1 out of bounds")
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
    """Vectorized angle calculation"""
    theta1 = np.arccos((L2**2 + L3**2 - L1**2) / (2 * L2 * L3))
    theta2 = np.arccos((L1**2 + L3**2 - L2**2) / (2 * L1 * L3))
    theta3 = np.arccos((L1**2 + L2**2 - L3**2) / (2 * L1 * L2))
    
    x1 = np.zeros_like(L1)
    x2 = np.pi - theta3
    x3 = 2*np.pi - (theta1 + theta3)
    return x1, x2, x3

def bin_N2(bin_edges, config, fold=False):
    N2_unnorm = 1 # Assume N2 input unnormalised is 1 for all inputs
    if fold == False:
        N_equi = N_bin(bin_edges, False) # Compute normalisation at bin level for equilateral case
        N2 = np.zeros(len(bin_edges) - 1)
        for i in range(len(bin_edges) - 1):
            L1_array = np.arange(bin_edges[i], bin_edges[i+1])
            L2_array = np.arange(bin_edges[i], bin_edges[i+1])
            L3_array = np.arange(bin_edges[i], bin_edges[i+1])

            # Get all combinations of L1, L2, L3
            L1_grid, L2_grid, L3_grid = np.meshgrid(L1_array, L2_array, L3_array)
            L1_flat = L1_grid.flatten()
            L2_flat = L2_grid.flatten()
            L3_flat = L3_grid.flatten()
            N_values = np.array([N(L1, L2, L3)*N2_unnorm for L1, L2, L3 in zip(L1_flat, L2_flat, L3_flat)])
            N2[i] = np.sum(N_values)
            N2[i] = N2[i]/N_equi[i]

        return N2
    
# Main code

def main():
    bin_edges = np.array([20,40, 60, 80, 100])#, 200, 400, 600, 800, 1000])
    N2 = bin_N2(bin_edges, config, fold=False)
    print('N2', N2)
    return 0

if __name__ == "__main__":
    main()
