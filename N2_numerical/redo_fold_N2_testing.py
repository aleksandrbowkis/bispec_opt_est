"""
    Computes the N2 bias to the reconstructed lensing bispectrum in the folded configuration. FOR PHI
"""

import numpy as np
from numpy import random 
from scipy.integrate import quad
from scipy.interpolate import interp1d
import sys
sys.path.append('/home/amb257/software/cmplx_cmblensplus/wrap')
sys.path.append('/home/amb257/software/cmplx_cmblensplus/utils')
# from cmblensplus/wrap/
import curvedsky as cs
from scipy.integrate import quad


################ Parameters ###############

lmax = 2000
Tcmb  = 2.726e6    # CMB temperature in microkelvin?
rlmin, rlmax = 2, 2000 # CMB multipole range for reconstruction
nside = 2048
bstype = 'fold'
nsims = 448 # Number of simulations to average over (in sets of 3) 
ellmin = 2 
ellmax = 2000 ##### 30/09/24 CHANGED TO 3000 CF SIMULATED RESULTS

################ Power spectra ################

ls, cl_unl, cl_len, cl_phi = np.loadtxt('/home/amb257/kappa_bispec/make_sims_parallel/camb_lencl_phi.txt')
L = np.arange(rlmax+1)
Lfac = (L*(L+1.) / 2 )**2
lcl = cl_len[0:rlmax+1] / Tcmb**2
ucl = cl_unl[0:rlmax+1] / Tcmb**2 #dimless unlensed T Cl
cl_kappa = Lfac * cl_phi[0:rlmax+1]
cl_phi = cl_phi[0:rlmax+1]

#Make noise power spectra
theta_fwhm = 1.4 #In arcminutes
sigma_noise = 10 #in muK-arcmin
arcmin2radfactor = np.pi / 60.0 / 180.0
noise_cl = (sigma_noise*arcmin2radfactor/Tcmb)**2*np.exp(L*(L+1.)*(theta_fwhm*arcmin2radfactor)**2/np.log(2.)/8.)
ocl = np.copy(lcl) + noise_cl #INC NOISE CF SIMULATED RESULTS NB EQUI NOISELESS ATM TO CF GIORGIO

# Interpolation functions for cl_kappa and ucl
cl_phi_interp = interp1d(L, cl_phi, kind='cubic', bounds_error=False, fill_value="extrapolate")
lcl_interp = interp1d(L, lcl, kind='cubic', bounds_error=False, fill_value="extrapolate")
ctot_interp = interp1d(L, ocl, kind='cubic', bounds_error=False, fill_value="extrapolate")
ctotprime = np.gradient(ocl, L)
ctotprime_interp = interp1d(L, ctotprime, kind='cubic', bounds_error=False, fill_value="extrapolate")
lclprime = np.gradient(lcl, L)
lclprime_interp = interp1d(L, lclprime, kind='cubic', bounds_error=False, fill_value="extrapolate")
lcldoubleprime = np.gradient(lclprime, L)
lcldoubleprime_interp = interp1d(L, lcldoubleprime, kind='cubic', bounds_error=False, fill_value="extrapolate")


def compute_n2_bias(l, A1, A2, L, pi):
    # Get interpolated values
    ctot = ctot_interp(l)
    ctt = lcl_interp(l)
    ctot_prime = ctotprime_interp(l)
    ctt_prime = lclprime_interp(l)
    ctt_double_prime = lcldoubleprime_interp(l)
    
    # Base factor (now including the additional l from spherical measure)
    base = (l/(256 * ctot**3)) * L**6 * pi
    
    # Rest of terms remain the same
    term1 = (-32 * A1 * ctot * ctt**2) + (80 * A2 * ctot * ctt**2)
    term2 = (64 * A1 * l * ctt**2 * ctot_prime) + (-16 * A2 * l * ctt**2 * ctot_prime)
    term3 = (-122 * A1 * l * ctot * ctt * ctt_prime) + (143 * A2 * l * ctot * ctt * ctt_prime)
    term4 = (72 * A1 * l**2 * ctt * ctot_prime * ctt_prime) + (-18 * A2 * l**2 * ctt * ctot_prime * ctt_prime)
    term5 = (-51 * A1 * l**2 * ctot * ctt_prime**2) + (69 * A2 * l**2 * ctot * ctt_prime**2)
    term6 = (20 * A1 * l**3 * ctot_prime * ctt_prime**2) + (-5 * A2 * l**3 * ctot_prime * ctt_prime**2)
    term7 = (-30 * A1 * l**2 * ctot * ctt * ctt_double_prime) + (21 * A2 * l**2 * ctot * ctt * ctt_double_prime)
    term8 = (-15 * A1 * l**3 * ctot * ctt_prime * ctt_double_prime) + (15 * A2 * l**3 * ctot * ctt_prime * ctt_double_prime)
    
    return base * (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8)

# Normalisation
phi_norm, phi_curl_norm = {}, {}
phi_norm['TT'], phi_curl_norm['TT'] = cs.norm_quad.qtt('lens',rlmax,rlmin,rlmax,lcl,ocl,lfac='')
norm_factor_phi = interp1d(L, phi_norm['TT'], kind='cubic', bounds_error=False, fill_value="extrapolate")

# Integration params
l_min, l_max = 2, 2000
pi_value = np.pi

# Array of L values to calculate integral for
lensingLarray = np.arange(2,1000,100)

# Array to hold integral
output = []

for L in lensingLarray:
    A1 = (1/(2*np.pi)**2)*norm_factor_phi(L) * cl_phi_interp(L/2)*cl_phi_interp(L/2)
    A2 = (1/(2*np.pi)**2)*norm_factor_phi(L/2) * cl_phi_interp(L/2) * cl_phi_interp(L)

    # Using scipy's quad for numerical integration
    result, error = quad(lambda l: compute_n2_bias(l, A1, A2, L, pi_value), l_min, l_max)
    output.append(result)

print(f"Integrated result: {output}")

# Save as text file 
np.savetxt('n2_bias_results.txt', (lensingLarray, output))