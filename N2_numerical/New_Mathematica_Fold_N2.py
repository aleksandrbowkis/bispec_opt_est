"""
    Computes the N2 bias to the reconstructed lensing bispectrum in the folded configuration. FOR PHI
    This is working with the latest expression from new mathematica notebook N2_testing.nb produced 14/1/25
    Use conda environment cmplx_fld_lensplus
    Very wrong output, must check expression and function, rewrite self.
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

l_min = 2
l_max = 3000
Tcmb  = 2.726e6    # CMB temperature in microkelvin?
rlmin, rlmax = 2, 3000 # CMB multipole range for reconstruction
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

################ Functions ################

def compute_n2_bias_folded(l, L, A_L, A_L_half):
    """
    Compute N2 bias for the folded configuration.
    
    Parameters:
    -----------
    l : float
        First multipole
    L : float
        Second multipole
    A_L : float
        A(L) phi normalisation
    A_L_half : float
        A(L/2) phi normalisation
    
    Returns:
    --------
    float : The computed N2 bias
    """
    # Get interpolated values
    Ctot = ctot_interp(l)
    Cpp_L = cl_phi_interp(L)
    Cpp_L_half = cl_phi_interp(L/2)
    Ctt = lcl_interp(l)
    dCtt = lclprime_interp(l)
    d2Ctt = lcldoubleprime_interp(l)
    dCtot = ctotprime_interp(l)
    
    # Compute the N2 bias
    prefactor = 1 / (1024*np.pi*Ctot**3)*l*(L*(L+1))**3*Cpp_L_half

    term1 = A_L*Cpp_L_half*(-4*l*dCtot*(16*Ctt**2 + 18*l*Ctt*dCtt + 5*l**2*dCtt**2) + Ctot*(32*Ctt**2 + 3*l**2*dCtt*(17*dCtt + 5*l*d2Ctt) + 2*l*Ctt*(61*dCtt + 15*l*d2Ctt)))
    
    term2 = A_L_half*Cpp_L*(-l*dCtot*(16*Ctt**2 + 18*l*Ctt*dCtt + 5*l**2*dCtt**2) + Ctot*(80*Ctt**2 + 3*l**2*dCtt*(23*dCtt + 5*l*d2Ctt) + l*Ctt*(143*dCtt + 21*l*d2Ctt)))

    result = prefactor*(term1 - term2)

    return result

################ Main code ################

# Normalisation
phi_norm, phi_curl_norm = {}, {}
phi_norm['TT'], phi_curl_norm['TT'] = cs.norm_quad.qtt('lens',rlmax,rlmin,rlmax,lcl,ocl,lfac='')
norm_factor_phi = interp1d(L, phi_norm['TT'], kind='cubic', bounds_error=False, fill_value="extrapolate")

# Array of L values to calculate integral for
lensingLarray = np.arange(2,1000,10)

# output hold integral
output = []

for L in lensingLarray:
    A_L = norm_factor_phi(L)
    A_L_half = norm_factor_phi(L/2)
    # Using scipy's quad for numerical integration
    result, error = quad(lambda l: compute_n2_bias_folded(l, L, A_L, A_L_half), l_min, l_max)
    output.append(result)

print(f"Integrated result: {output}")

# Save as text file 
np.savetxt('new_mathematica_N2_fold.txt', (lensingLarray, output))
print(f'done l_max: {l_max} ellmax: {ellmax}')