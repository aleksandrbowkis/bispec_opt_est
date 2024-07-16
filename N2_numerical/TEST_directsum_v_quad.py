import numpy as np
from numpy import random 
from scipy.integrate import quad
from scipy.interpolate import interp1d


################ Parameters ###############

lmax = 2000
Tcmb  = 2.726e6    # CMB temperature in microkelvin?
rlmin, rlmax = 2, 2000 # CMB multipole range for reconstruction
nside = 2048
bstype = 'equi'
nsims = 448 # Number of simulations to average over (in sets of 3) 
ellmin = 2 
ellmax = 2000 ##### check!!!! vs sims

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
ocl = np.copy(lcl) + noise_cl

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
def integrand_fn(lensingL, ell, cl_phi_interp, lcl_interp, ctot_interp, ctotprime_interp, lclprime_interp, lcldoubleprime_interp):

    integrand = (2 * np.pi) ** 2 * cl_phi_interp(lensingL) ** 2 * 1 / (32 * ctot_interp(ell) ** 3) * lensingL ** 6 * np.pi * (
        -ell * ctotprime_interp(ell) * (8 * lcl_interp(ell) ** 2 + 6 * ell * lcl_interp(ell) * lclprime_interp(ell) + ell ** 2 * lclprime_interp(ell) ** 2) 
        + ctot_interp(ell) * (32 * lcl_interp(ell) ** 2 + 6 * ell ** 2 * lclprime_interp(ell) ** 2 + ell * lcl_interp(ell) * (41 * lclprime_interp(ell) + 3 * ell * lcldoubleprime_interp(ell)))
    )
    return integrand

lensingLarray = np.arange(1,100,10)
output_quad = []
output_direct = []

for lensingL in lensingLarray:
    # Wrapper function for quad
    def integrand_wrapper(ell):
        return integrand_fn(lensingL, ell, cl_phi_interp, lcl_interp, ctot_interp, ctotprime_interp, lclprime_interp, lcldoubleprime_interp)

    integrand_values = []
    for ell in np.arange(ellmin, ellmax + 1):
        integrand = integrand_fn(lensingL, ell, cl_phi_interp, lcl_interp, ctot_interp, ctotprime_interp, lclprime_interp, lcldoubleprime_interp)
        integrand_values.append(integrand)

    # Now write some code to integrate this using quad
    integral_quad, error = quad(integrand_wrapper, ellmin, ellmax, limit=100)

    # Now compute the same integrand using direct summation
    integral_direct_sum = np.sum(integrand_values)

    print('quad', integral_quad)
    print('direct sum', integral_direct_sum)

    output_quad.append(integral_quad)
    output_direct.append(integral_direct_sum)

#change outputs to numpy arrays
output_quad = np.array(output_quad)
output_direct = np.array(output_direct)

print(lensingLarray)
print(output_quad)

np.savetxt('TEST_quad_equi.txt', (lensingLarray, output_quad))
np.savetxt('TEST_direct_equi.txt', (lensingLarray, output_direct))