"""
    Computes the N2 bias to the reconstructed lensing bispectrum in the folded configuration.
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
ellmax = 3000 ##### 30/09/24 CHANGED TO 3000 CF SIMULATED RESULTS

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

# Define a the integrand
# Note cf mathematica we add a factor of QE normalisation,1/(2pi)^2 from integral not expanded 
def integrand_fn(lensingL, ell, cl_phi_interp, lcl_interp, ctot_interp, ctotprime_interp, lclprime_interp, lcldoubleprime_interp):

    integrand = (1/(2*np.pi)**2) * cl_phi_interp(lensingL) ** 2 * 1 / (256 * ctot_interp(ell) ** 3) * 3 * lensingL ** 3 * (lensingL+1)**3  * np.pi * (
        16 * ctot_interp(ell) * lcl_interp(ell) ** 2 + 16 * ell * lcl_interp(ell)**2 * ctotprime_interp(ell) + 7 * ell * ctot_interp(ell) * lcl_interp(ell) * lclprime_interp(ell) + 
         18 * ell**2 * lcl_interp(ell) * ctotprime_interp(ell) * lclprime_interp(ell) + 6 * ell**2 * ctot_interp(ell) * lclprime_interp(ell)**2 +
         5 * ell**3 * ctotprime_interp(ell) * lclprime_interp(ell)**2 - 3 * ell**2 * ctot_interp(ell) * lcl_interp(ell) * lcldoubleprime_interp(ell)
    )
    return integrand

lensingLarray = np.arange(1,1000,10)
output_quad = []
output_direct = []

# Now calculate the normalisation. Outside loop as unnecessary to repeatedly calculate.
phi_norm, phi_curl_norm = {}, {}
phi_norm['TT'], phi_curl_norm['TT'] = cs.norm_quad.qtt('lens',lmax,rlmin,rlmax,lcl,ocl,lfac='')

for lensingL in lensingLarray:
    # Wrapper function for quad, call integrand_fn for every lensingL
    def integrand_wrapper(ell):
        return integrand_fn(lensingL, ell, cl_phi_interp, lcl_interp, ctot_interp, ctotprime_interp, lclprime_interp, lcldoubleprime_interp)

    # Populate the integrand for this lensing L at every cmb l up to 2000
    integrand_values = []
    for ell in np.arange(ellmin, ellmax + 1):
        integrand = integrand_fn(lensingL, ell, cl_phi_interp, lcl_interp, ctot_interp, ctotprime_interp, lclprime_interp, lcldoubleprime_interp)
        integrand_values.append(integrand)

    # Now write some code to integrate this using quad
    integral_quad, error = quad(integrand_wrapper, ellmin, ellmax, limit=100)

    # Now compute the same integrand using direct summation
    integral_direct_sum = np.sum(integrand_values)

    # Now calculate the normalisation for this lensingL. Recall we're using phi everywhere as the low L N2 result was computed for phi.
    norm_factor_phi = phi_norm['TT'][int(lensingL)]

    # Normalise the two integrals
    integral_quad *= norm_factor_phi
    integral_direct_sum *= norm_factor_phi

    print('quad', integral_quad)
    print('direct sum', integral_direct_sum)
    print('norm_factor_phi', norm_factor_phi)

    output_quad.append(integral_quad)
    output_direct.append(integral_direct_sum)

# Convert to kappa bias from phi
output_quad *= (0.5*lensingLarray*(lensingLarray+1))**3
output_direct *= (0.5*lensingLarray*(lensingLarray+1))**3

# Change outputs to numpy arrays
output_quad = np.array(output_quad)
output_direct = np.array(output_direct)

# Save outputs
np.savetxt('fix_TEST_quad_fold.txt', (lensingLarray, output_quad))
np.savetxt('TEST_direct_fold.txt', (lensingLarray, output_direct))

