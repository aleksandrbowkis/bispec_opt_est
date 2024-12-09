"""
    Computes the N2 bias to the reconstructed lensing bispectrum. Can then be called for specific triangle configurations.
    Used in investigation of binning effect at low multipoles
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

###################### Integrand ###########################

def integrand_fn(ell, cl_phi_interp, lcl_interp, ctot_interp, ctotprime_interp, lclprime_interp, lcldoubleprime_interp, norm_factor_phi, x1, x2, x3, L1, L2, L3):
    """ Integrand for series approximation of N2 at low multipoles. x1,2,3 are the internal angles of triangle with side length L1,2,3 defined from positive x axis. """
    A1 = (1/(2*np.pi)**2)*norm_factor_phi(L1) * cl_phi_interp(L2)*cl_phi_interp(L3)
    A2 = (1/(2*np.pi)**2)*norm_factor_phi(L2) * cl_phi_interp(L1)*cl_phi_interp(L3)
    A3 = (1/(2*np.pi)**2)*norm_factor_phi(L3) * cl_phi_interp(L2)*cl_phi_interp(L1)

    integrand = 1 / (32 * ctot_interp(ell) ** 3)*L1*L2*L3**2*np.pi*(2*ell*ctotprime_interp(ell)*(8*(3*(A1*L1**2+A2*L2**2)*np.cos(x1-x2)+(A1*L1**2+A2*L2**2)*Cos(x1+x2-2*x3) +
                                                                                                    A3*L1*L3*(np.cos(2*x1-x2-x3)+3*np.cos(x2-x3)))*lcl_interp(ell)**2+2*ell*(13*(A1*L1**2+A2*L2**2)*np.cos(x1-x2)+5*(A1*L1**2+A2*L2**2)*np.cos(x1+x2-2*x3)+A3*L1*L3*(5*np.cos(2*x1-x2-x3)+13*np.cos(x2-x3)))*lcl_interp(ell)*lclprime_interp(ell) + 
                                                                                                                                                                             ell**2*(6*(A1*L1**2+A2*L2**2)*np.cos(x1-x2)+A3*L1*L3*np.cos(2*x1+x2-3*x3)+A1*L1**2*np.cos(3*x1-x2-2*x3)+3*A1*L1**2*np.cos(x1+x2-2*x3)+3*A2*L2**2*np.cos(x1+x2-2*x3)+3*A3*L1*L3*np.cos(2*x1-x2-x3)+6*A3*L1*L3*np.cos(x2-x3)+A2*L2**2*np.cos(x1-3*x2+2*x3))*lclprime_interp(ell)**2)+
                                                                                                                                                                             ctot_interp(ell)*(64*(A3*L1**2*np.cos(x1-x2)+A2*L2*L3*np.cos(x1-x3)+A1*L1*L3*np.cos(x2-x3))*lcl_interp(ell)**2-2*ell*lcl_interp(ell)*(((27*A1*L1**2 - 50*A3*L1**2+27*A2*L2**2)*np.cos(x1-x2)+9*(A1*La**2+A2*L2**2)*np.cos(x1+x2-2*x3)+
                                                                                                                                                                                                                                                                                                                    L3*(-50*A2*L2*np.cos(x1-x3)+9*A3*L1*np.cos(2*x1-x2-x3)+(-50*A1+27*A3)*L1*np.cos(x2-x3)))*lclprime_interp(ell)+
                                                                                                                                                                                                                                                                                                                    3*ell*((3*A1*L1**2-2*A3*L1**2+3*A2*L2**2)*np.cos(x1-x2)+(A1*L1**2+A2*L2**2)*np.cos(x1+x2-2*x3)+L3*(-2*A2*L2*np.cos(x1-x3)+A3*L1*np.cos(2*x1-x2-x3)+(-2*A1+3*A3)*L1*np.cos(x2-x3)))*lcldoubleprime_interp(ell))+
                                                                                                                                                                                                                                                                                                                    ell**2*lclprime_interp(ell)*((-2*(9*A1*L1**2-17*A3*L1**2+9*A2*L2**2)*np.cos(x1-x2)+(3*A1+A3)*L1*L3*np.cos(2*x1+x2-3*x3)+3*A2*L2*L3*np.cos(x1+2*x2-3*x3)+A1*L1**2*np.cos(3*x1-x2-2*x3)+3*A3*L1**2*np.cos(3*x1-x2-2*x3)-
                                                                                                                                                                                                                                                                                                                                                  9*A1*L1**2*np.cos(x1+x2-2*x3)+13*A3*L1**2*np.cos(x1+x2-2*x3)-9*A2*L2**2*np.cos(x1+x2-2*x3)+34*A2*L2*L3*np.cos(x1-x3)+13*A1*L1*L3*np.cos(2*x1-x2-x3)-9*A3*L1*L3*np.cos(2*x1-x2-x3)+
                                                                                                                                                                                                                                                                                                                                                  34*a1*L1*L3*np.cos(x2-x3)-18*A3*L1*L3*np.cos(x2-x3)+13*A2*L2*L3*np.cos(x1-2*x2+x3)+A2*L2**2*np.cos(x1-3*x2+2*x3))*lclprime_interp(ell)+ell*(-6*(A1*L1**2-A3*L1**2+A2*L2**2)*np.cos(x1-x2)+(A1-A3)*L1*L3*np.cos(2*x1+x2-3*x3)+
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              A2*L2*L3*np.cos(x1+2*x2-3*x3)-A1*L1**2*np.cos(3*x1-x2-2*x3)+A3*L1**2*np.cos(3*x1-x2-2*x3)-
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              3*A1*L1**2*np.cos(x1+x2-2*x3)+3*A3*L1**2*np.cos(x1+x2-2*x3)-3*A2*L2**2*np.cos(x1+x2-2*x3)+
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              6*A2*L2*L3*np.cos(x1-x3)+3*A1*L1*L3*np.cos(2*x1-x2-x3)-3*A3*L1*L3*np.cos(2*x1-x2-x3)+6*A1*L1*L3*np.cos(x2-x3)-
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              6*A3*L1*L3*np.cos(x2-x3)+3*A2*L2*L3*np.cos(x1-2*x2+x3)-A2*L2**2*np.cos(x1-3*x2+2*x3))*lcldoubleprime_interp(ell))))
    return integrand