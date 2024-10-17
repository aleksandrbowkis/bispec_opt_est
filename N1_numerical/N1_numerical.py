""" code to calculate the N1 bias to the reconstructed lensing bispectrum"""

import numpy as np
import vegas
from scipy.integrate import quad
from scipy.interpolate import interp1d
import sys, os
sys.path.append('/home/amb257/software/cmplx_cmblensplus/wrap')
sys.path.append('/home/amb257/software/cmplx_cmblensplus/utils')
# from cmblensplus/wrap/
import curvedsky as cs
import flatsky as fs
import multiprocessing as mp

# Parameters
lmax = 2000
Tcmb  = 2.726e6    # CMB temperature in microkelvin?
bstype = 'equi'
ellmin, ellmax = 2, 2000

# Define the directory where power spec are stored
input_dir = "../Power_spectra"

# Load the power spec
L = np.arange(0,2000+1,1)
ucl = np.loadtxt(os.path.join(input_dir, "unlensed_clTT_lmax8000.txt"))
gcl = np.loadtxt(os.path.join(input_dir, "glensed_clTT_lmax8000.txt"))
lcl = np.loadtxt(os.path.join(input_dir, "lensed_clTT_lmax8000.txt"))
ucl = ucl[0:2001]
gcl = gcl[0:2001]
lcl = lcl[0:2001]

#Make noise power spectra
theta_fwhm = 1.4 #In arcminutes
sigma_noise = 10 #in muK-arcmin
arcmin2radfactor = np.pi / 60.0 / 180.0
noise_cl = (sigma_noise*arcmin2radfactor/Tcmb)**2*np.exp(L*(L+1.)*(theta_fwhm*arcmin2radfactor)**2/np.log(2.)/8.)
ctot = np.copy(lcl) + noise_cl

# Now interpolate them
#cl_phi_interp = interp1d(L, cl_phi, kind='cubic', bounds_error=False, fill_value="extrapolate")
ucl_interp = interp1d(L, ucl, kind='cubic', bounds_error=False, fill_value="extrapolate")
lcl_interp = interp1d(L, lcl, kind='cubic', bounds_error=False, fill_value="extrapolate")
gcl_interp = interp1d(L, gcl, kind='cubic', bounds_error=False, fill_value="extrapolate")
ctot_interp = interp1d(L, ctot, kind='cubic', bounds_error=False, fill_value="extrapolate")

# Load the normalisation
flat_sky_norm = np.load("/home/amb257/kappa_bispec/bispec_opt_est/N0_numerical/normalisation/flat_sky_norm.npy")
print('shape', np.shape(flat_sky_norm))
flat_sky_norm_interp = interp1d(L, flat_sky_norm, kind='cubic', bounds_error=False, fill_value="extrapolate")