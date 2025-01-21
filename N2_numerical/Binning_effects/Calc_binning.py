""" Code to compute the permissible triangles within a multipole bin 
then compute N2 (series approximation) for them to investigate the binning effects at low multipoles"""

import numpy as np
import sys
sys.path.append('/home/amb257/software/cmplx_cmblensplus/wrap')
sys.path.append('/home/amb257/software/cmplx_cmblensplus/utils')
# from cmblensplus/wrap/
import curvedsky as cs

from full_N2_series_integral import calculate_N2

###### Functions ######
def find_triangles(bin_min, bin_max):
    L = np.arange(bin_min, bin_max + 1)
    L1, L2, L3 = np.meshgrid(L, L, L, indexing='ij') # Creates a coordinate grid of all possible L1,2,3 values
    mask = (L1 + L2 > L3) & (L2 + L3 > L1) & (L3 + L1 > L2) & (L1 <= L2) & (L2 <= L3) #Implement triangle inequality. Note requiring L1 <= L2 etc means we dont double count 3,4,5 and 4,3,5 etc. 
    valid_triangles = np.array([L1[mask], L2[mask], L3[mask]]).T # Mask out disallowed triangles. Transpose st each row represents a different triangle.
    return valid_triangles

def find_angles(L1, L2, L3):
    # Cosine rule gives interior angles.
    theta1 = np.arccos((L2**2 + L3**2 - L1**2) / (2 * L2 * L3))
    theta2 = np.arccos((L1**2 + L3**2 - L2**2) / (2 * L1 * L3))
    theta3 = np.arccos((L1**2 + L2**2 - L3**2) / (2 * L1 * L2))

    # All angles calculated from x axis anticlockwise. L1 aligned with x axis then continues anticlockwise to L2 then L3
    x1 = 0
    x2 = np.pi - theta3
    x3 = 2*np.pi - (theta1+theta3)
    return x1, x2, x3


####### Tests ######

L1 = 100
L2 = L1 
L3 = L1
x1 = 0
x2 = 2*np.pi/3
x3 = 4*np.pi/3

output = calculate_N2(L1, L2, L3, x1, x2, x3)
print(output)

what_are_my_triangles = find_triangles(2,5)

x1,x2,x3 = find_angles(2, 2, 2)
print(x1, x2, x3)

################ Main ############

# Now define binning scheme. Note this only up to L of 100 atm
bin_edges = [20,40,60,80,100] #200,300] #400,500, 600, 700, 800, 900, 1000]
bin_edges = np.array(bin_edges)
bin_mid = 0.5 * (bin_edges[1:] + bin_edges[:-1])
averaged_N2_bin = []

for i in range(len(bin_edges) - 1):
    # Find the bin edges
    bin_min = bin_edges[i]
    bin_max = bin_edges[i + 1]
    which_triangles = find_triangles(bin_min, bin_max)

    N2_bin = []

    for triangle in which_triangles:
        L1, L2, L3 = triangle
        x1, x2, x3 = find_angles(L1, L2, L3)
        N2_value = calculate_N2(L1, L2, L3, x1, x2, x3)
        N2_bin.append(N2_value)

    average_N2_in_this_bin = np.mean(N2_bin)
    averaged_N2_bin.append(average_N2_in_this_bin)

# Save
np.save('binning_tests_outputs/binned_equilateral.npy', (bin_mid, averaged_N2_bin))
