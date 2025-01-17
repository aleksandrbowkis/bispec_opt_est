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

# Define a the integrand
# Note cf mathematica we add a factor of QE normalisation,1/(2pi)^2 from integral not expanded and the cL/2 phi (recall these L/2 in folded case)

# We now use new mathematica expression where AL, AL/2 and AL/3 are used for three different terms. Note that the phi power sepctr are calculated at the other two lensing L's 
# in the folded configuration that don't appear in the normalisation factor. A1,2,3 in the mathematica notebook are the norm and the appropriate Clphi's




def integrand_fn(lensingL, ell, cl_phi_interp, lcl_interp, ctot_interp, ctotprime_interp, lclprime_interp, lcldoubleprime_interp, norm_factor_phi):

    # integrand = (1/(2*np.pi)**2)*norm_factor_phi(lensingL) * cl_phi_interp(lensingL) ** 2 * 1 / (256 * ctot_interp(ell) ** 3) * 3 * lensingL ** 3 * (lensingL+1)**3 * np.pi * ell * (-16*ctot_interp(ell)*lcl_interp(ell)**2 - 16*ell*lcl_interp(ell)**2*ctotprime_interp(ell) - 7*ell*ctot_interp(ell)*lcl_interp(ell)*lclprime_interp(ell) -
    #     18*ell**2*lcl_interp(ell)*ctotprime_interp(ell)*lclprime_interp(ell) - 6*ell**2*ctot_interp(ell)*lclprime_interp(ell)**2 - 5*ell**3*ctotprime_interp(ell)*lclprime_interp(ell)**2 + 3*ell**2*ctot_interp(ell)*lcl_interp(ell)*lcldoubleprime_interp(ell))
    # return integrand
    #The below changed each normalisation factor but this gives best results when all factors are the same so perhaps not correct. Reverting to previous equal norm expression.
    A1 = (1/(2*np.pi)**2)*norm_factor_phi(lensingL) * cl_phi_interp(lensingL/2)*cl_phi_interp(lensingL/2)
    A2 = (1/(2*np.pi)**2)*norm_factor_phi(lensingL/2) * cl_phi_interp(lensingL/2) * cl_phi_interp(lensingL)
    #A3 = A2

    #FORGOT THE ELL FACTOR FOR MEASURE. Note this (2/12/24) looks much better with the same normalisation factor for each so may have been wrong... But still not quite right (approx factor of 10 too small)
    # Changing the clphi to L/2 boosts the integral.
    F1 = (-32 * ctot_interp(ell) * lcl_interp(ell)**2
      + 64 * ell * ctotprime_interp(ell) * lcl_interp(ell)**2
      - 122 * ell * ctot_interp(ell) * lcl_interp(ell) * lclprime_interp(ell)
      + 72 * ell**2 * lcl_interp(ell) * ctotprime_interp(ell) * lclprime_interp(ell)
      - 51 * ell**2 * ctot_interp(ell) * lclprime_interp(ell)**2
      + 20 * ell**3 * ctotprime_interp(ell) * lclprime_interp(ell)**2
      - 30 * ell**2 * ctot_interp(ell) * lcl_interp(ell) * lcldoubleprime_interp(ell)
      - 15 * ell**3 * ctot_interp(ell) * lclprime_interp(ell) * lcldoubleprime_interp(ell))

    F2 = (80 * ctot_interp(ell) * lcl_interp(ell)**2
      - 16 * ell * lcl_interp(ell)**2 * ctotprime_interp(ell)
      + 143 * ell * ctot_interp(ell) * lcl_interp(ell) * lclprime_interp(ell)
      - 18 * ell**2 * lcl_interp(ell) * ctotprime_interp(ell) * lclprime_interp(ell)
      + 69 * ell**2 * ctot_interp(ell) * lclprime_interp(ell)**2
      - 5 * ell**3 * ctotprime_interp(ell) * lclprime_interp(ell)**2
      + 21 * ell**2 * ctot_interp(ell) * lcl_interp(ell) * lcldoubleprime_interp(ell)
      + 15 * ell**3 * ctot_interp(ell) * lclprime_interp(ell) * lcldoubleprime_interp(ell))

    integrand = (1 / (256 * ctot_interp(ell)**3)
             * lensingL**3 * (lensingL + 1)**3 * np.pi * ell
             * (A1 * F1 + A2 * F2))

    # integrand = 1 / (256 * ctot_interp(ell) ** 3) * lensingL ** 3 * (lensingL+1)**3 * np.pi * ell * (-32*A1*ctot_interp(ell)*lcl_interp(ell)**2 + 80*A2*ctot_interp(ell)*lcl_interp(ell)**2 + 64*A1*ell*ctotprime_interp(ell)*lcl_interp(ell)**2 - 
    #                                                                                                   16*A2*ell*lcl_interp(ell)**2*ctotprime_interp(ell) - 122*A1*ell*ctot_interp(ell)*lcl_interp(ell)*lclprime_interp(ell) + 143*A2*ell*ctot_interp(ell)*lcl_interp(ell)*lclprime_interp(ell) + 
    #                                                                                                   72*A1*ell**2*lcl_interp(ell)*ctotprime_interp(ell)*lclprime_interp(ell) - 18*A2*ell**2*lcl_interp(ell)*ctotprime_interp(ell)*lclprime_interp(ell) - 51*A1*ell**2*ctot_interp(ell)*lclprime_interp(ell)**2 + 
    #                                                                                                   69*A2*ell**2*ctot_interp(ell)*lclprime_interp(ell)**2 + 20*A1*ell**3*ctotprime_interp(ell)*lclprime_interp(ell)**2 - 5*A2*ell**3*ctotprime_interp(ell)*lclprime_interp(ell)**2 - 30*A1*ell**2*ctot_interp(ell)*lcl_interp(ell)*lcldoubleprime_interp(ell) + 
    #                                                                                                   21*A2*ell**2*ctot_interp(ell)*lcl_interp(ell)*lcldoubleprime_interp(ell) - 15*A1*ell**3*ctot_interp(ell)*lclprime_interp(ell)*lcldoubleprime_interp(ell) + 15*A2*ell**3*ctot_interp(ell)*lclprime_interp(ell)*lcldoubleprime_interp(ell))                      

    # integrand = 1 / (256 * ctot_interp(ell) ** 3) * lensingL ** 3 * (lensingL+1)**3 * np.pi * ell * (32*A1*ctot_interp(ell)*lcl_interp(ell)**2 - 16*A2*ctot_interp(ell)*lcl_interp(ell)**2 - 64*A3*ctot_interp(ell)*lcl_interp(ell)**2 - 
    #     64*A1*ell*lcl_interp(ell)**2*ctotprime_interp(ell) - 16*A2*ell*lcl_interp(ell)**2*ctotprime_interp(ell) +32*A3*ell*lcl_interp(ell)**2*ctotprime_interp(ell) + 122*A1*ell*ctot_interp(ell)*lcl_interp(ell)*lclprime_interp(ell) - 
    #     7*A2*ell*ctot_interp(ell)*lcl_interp(ell)*lclprime_interp(ell) - 136*A3*ell*ctot_interp(ell)*lcl_interp(ell)*lclprime_interp(ell) - 72*A1*ell**2*lcl_interp(ell)*ctotprime_interp(ell)*lclprime_interp(ell) - 
    #     18*A2*ell**2*lcl_interp(ell)*ctotprime_interp(ell)*lclprime_interp(ell) + 36*A3*ell**2*lcl_interp(ell)*ctotprime_interp(ell)*lclprime_interp(ell) + 51*A1*ell**2*ctot_interp(ell)*lclprime_interp(ell)**2 - 
    #     6*A2*ell**2*ctot_interp(ell)*lclprime_interp(ell)**2 - 63*A3*ell**2*ctot_interp(ell)*lclprime_interp(ell)**2 - 20*A1*ell**3*ctotprime_interp(ell)*lclprime_interp(ell)**2 - 5*A2*ell**3*ctotprime_interp(ell)*lclprime_interp(ell)**2 + 
    #     10*A3*ell**3*ctotprime_interp(ell)*lclprime_interp(ell)**2 + 30*A1*ell**2*ctot_interp(ell)*lcl_interp(ell)*lcldoubleprime_interp(ell) + 3*A2*ell**2*ctot_interp(ell)*lcl_interp(ell)*lcldoubleprime_interp(ell) - 
    #     24*A3*ell**2*ctot_interp(ell)*lcl_interp(ell)*lcldoubleprime_interp(ell) + 15*A1*ell**3*ctot_interp(ell)*lclprime_interp(ell)*lcldoubleprime_interp(ell) - 15*A3*ell**3*ctot_interp(ell)*lclprime_interp(ell)*lcldoubleprime_interp(ell))
    return integrand

lensingLarray = np.arange(2,1000,10)
output_quad = []
output_direct = []

# Now calculate the normalisation. Outside loop as unnecessary to repeatedly calculate.
phi_norm, phi_curl_norm = {}, {}
phi_norm['TT'], phi_curl_norm['TT'] = cs.norm_quad.qtt('lens',rlmax,rlmin,rlmax,lcl,ocl,lfac='')
print(L)
norm_factor_phi = interp1d(L, phi_norm['TT'], kind='cubic', bounds_error=False, fill_value="extrapolate")

for lensingL in lensingLarray:
    # Wrapper function for quad, call integrand_fn for every lensingL
    def integrand_wrapper(ell):
        return integrand_fn(lensingL, ell, cl_phi_interp, lcl_interp, ctot_interp, ctotprime_interp, lclprime_interp, lcldoubleprime_interp, norm_factor_phi)

    # Populate the integrand for this lensing L at every cmb l up to 2000
    integrand_values = []
    for ell in np.arange(ellmin, ellmax + 1):
        integrand = integrand_fn(lensingL, ell, cl_phi_interp, lcl_interp, ctot_interp, ctotprime_interp, lclprime_interp, lcldoubleprime_interp, norm_factor_phi)
        integrand_values.append(integrand)

    # Now write some code to integrate this using quad
    integral_quad, error = quad(integrand_wrapper, ellmin, ellmax, limit=100)

    # Now compute the same integrand using direct summation
    integral_direct_sum = np.sum(integrand_values)

    print('quad', integral_quad)
    print('direct sum', integral_direct_sum)

    output_quad.append(integral_quad)
    output_direct.append(integral_direct_sum)


# Change outputs to numpy arrays
output_quad = np.array(output_quad)
output_direct = np.array(output_direct)

print('CLphi50', cl_phi[50])
print('phi norm 100', phi_norm['TT'][100])

# Save outputs
np.savetxt('N2_fold.txt', (lensingLarray, output_quad))
#np.savetxt('allAs_same_norm_direct_fold.txt', (lensingLarray, output_direct))


