""" Code to compute the permissible triangles within a multipole bin 
then compute N2 (series approximation) for them to investigate the binning effects at low multipoles"""

import numpy as np
import sys
sys.path.append('/home/amb257/software/cmplx_cmblensplus/wrap')
sys.path.append('/home/amb257/software/cmplx_cmblensplus/utils')
# from cmblensplus/wrap/
import curvedsky as cs

from full_N2_series_integral import calculate_N2

####### Main ######

L1 = 100
L2 = L1 
L3 = L1
x1 = 0
x2 = 2*np.pi/3
x3 = 4*np.pi/3

output = calculate_N2(L1, L2, L3, x1, x2, x3)
print(output)

# Now calculate all permissible triangles within a multipole bin
