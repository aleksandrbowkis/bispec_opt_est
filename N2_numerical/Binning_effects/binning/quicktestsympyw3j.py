import sympy as sp
from sympy.physics.quantum.cg import Wigner3j
import numpy as np
import sys, os
sys.path.append('/home/amb257/software/cmplx_cmblensplus/wrap')
sys.path.append('/home/amb257/software/cmplx_cmblensplus/utils')
import curvedsky as cs
import basic

# Calculate a Wigner 3j symbol
# Format: Wigner3j(j1, j2, j3, m1, m2, m3)
# result = Wigner3j(1, 1, 2, 0, 0, 0)# Evaluate to a numerical value if needed
# numerical_result = float(result.doit())
# print(numerical_result)

numerical_result = float(Wigner3j(3, 4, 5, 0, 0, 0).doit())
#print(numerical_result)

def N_sympy_bin(bin_edges, is_it_folded):
    """ Compute the normalisation factor for all bins using sympy"""
    size_bin_edges = len(bin_edges)
    N = np.zeros(size_bin_edges-1)

    if is_it_folded == False:
        for index, item in enumerate(bin_edges[0:size_bin_edges-1]):
            lower_bound_bin = int(item)
            upper_bound_bin = int(bin_edges[index+1])
            for L1 in range(int(lower_bound_bin), int(upper_bound_bin)):
                for L2 in range(int(lower_bound_bin), int(upper_bound_bin)):
                    for L3 in range(int(lower_bound_bin), int(upper_bound_bin)):
                        w3j = float(Wigner3j(L1, L2, L3, 0, 0, 0).doit())
                        #print(w3j)
                        N[index] += (2*L1+1)*(2*L2+1)*(2*L3+1) / (4*np.pi) * w3j**2
    else:
        for index, item in enumerate(bin_edges[0:size_bin_edges-1]):
            lower_bound_bin = int(item)
            upper_bound_bin = int(bin_edges[index+1])
            for L1 in range(int(lower_bound_bin), int(upper_bound_bin)):
                for L2 in range(int(lower_bound_bin/2), int(upper_bound_bin/2)):
                    for L3 in range(int(lower_bound_bin/2), int(upper_bound_bin/2)):
                        w3j = float(Wigner3j(L1, L2, L3, 0, 0, 0).doit())
                        N[index] += (2*L1+1)*(2*L2+1)*(2*L3+1) / (4*np.pi) * w3j**2
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
                print('l3,l2', l3, l2)
                lower_bound_w3j = np.abs(l3 - l2)
                print('lower bound', lower_bound_w3j)
                upper_bound_w3j = l3 + l2
                #Calculate the w3j's
                w3j = basic.wigner_funcs.wigner_3j(l3,l2,0,0)
                for l1 in range(lower_bound_bin, upper_bound_bin):
                    if l1 >= lower_bound_w3j and l1 <= upper_bound_w3j:
                        print('l1', l1)
                        position_l1_in_w3j = l1 - lower_bound_w3j #this is the position of the current value of l1 in the w3j array
                        print('pos', position_l1_in_w3j)
                        sum += (2*l1+1)*(2*l2+1)*(2*l3+1) * w3j[position_l1_in_w3j]**2 / (4*np.pi)
        N[index] = sum
    return N

print(N_bin([2,4],False))
# print(N_sympy_bin([2,4],False))

# w3j = basic.wigner_funcs.wigner_3j(3,4,0,0)
# print(w3j)