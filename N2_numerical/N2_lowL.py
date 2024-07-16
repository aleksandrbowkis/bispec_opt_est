"""
Code to compute the low (lensing) L approximation (l used for cmb multipoles) to the lensing bispectrum for comparison to the optimal estimator.
"""


#### Import modules
import numpy as np
import vegas 
from scipy.interpolate import interp1d
import healpy as hp
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
cl_kappa_interp = interp1d(L, cl_kappa, kind='cubic', bounds_error=False, fill_value="extrapolate")
ucl_interp = interp1d(L, ucl, kind='cubic', bounds_error=False, fill_value="extrapolate")
lcl_interp = interp1d(L, lcl, kind='cubic', bounds_error=False, fill_value="extrapolate")
ctot_interp = interp1d(L, ocl, kind='cubic', bounds_error=False, fill_value="extrapolate")
ctotprime = np.gradient(ocl, L)
ctotprime_interp = interp1d(L, ctotprime, kind='cubic', bounds_error=False, fill_value="extrapolate")
uclprime = np.gradient(ucl, L)
uclprime_interp = interp1d(L, uclprime, kind='cubic', bounds_error=False, fill_value="extrapolate")
ucldoubleprime = np.gradient(uclprime, L)
ucldoubleprime_interp = interp1d(L, ucldoubleprime, kind='cubic', bounds_error=False, fill_value="extrapolate")
lclprime = np.gradient(lcl, L)
lclprime_interp = interp1d(L, lclprime, kind='cubic', bounds_error=False, fill_value="extrapolate")
lcldoubleprime = np.gradient(lclprime, L)
lcldoubleprime_interp = interp1d(L, lcldoubleprime, kind='cubic', bounds_error=False, fill_value="extrapolate")

################# Functions ##################

def dotprod(l1,l2):
    #Computes the dot product of two (multipole) vectors.
    l1dotl2 = l1[0]*l2[0] + l1[1]*l2[1]
    return l1dotl2

def vect_modulus(l1):
    #Computes the modulus of an input vector
    modl1 = np.sqrt(l1[0]**2 + l1[1]**2)
    return modl1

def response_func(CTlens, l1, l2, sizel1, sizel2):
    #Computes the response function f where <T(l1)T(l2)> = f(l1,l2)phi (T here is the lensed CMB temperature, phi is the lensing potential). Comes from taylor expanding the lensed temp.
    #CTlens is lensed power spectrum, l1, l2 are the multipole values.
    #L is the multipole at which we will evaluate the N(0) bias term
    L = l1 + l2
    f12 = dotprod(L,l1)*CTlens(sizel1) + dotprod(L,l2)*CTlens(sizel2)
    return f12

def bigF(l1, l2, l1size, l2size, CTlens, Ctotal):
    #computes the function F(l1,l2) = f(l1,l2)/(2 Ctot(l1) Ctot(l2))
    # f12 is the response function
    
    bigF = response_func(CTlens, l1, l2, l1size, l2size) / ( 2 * Ctotal(l1size) * Ctotal(l2size))
    return bigF

def make_equilateral_L(sizeL):
    # Makes a set of 3 L's satisfying the triangle condition w/ L1 along the x axis and each L same length.
    L1, L2, L3 = np.zeros(2), np.zeros(2), np.zeros(2)
    L1[0] = sizeL
    
    L2[0] = -sizeL * np.cos(np.pi/3)
    L2[1] = sizeL * np.sin(np.pi/3)

    L3[0] = -sizeL * np.cos(np.pi/3)
    L3[1] = -sizeL * np.sin(np.pi/3)
    
    return L1, L2, L3

def make_fold_L(sizeL):
    # Makes a set of 3 L's satisfying the condition L1 = 2L2 = 2L3 satisfying the triangle condition.
    L1, L2, L3 = np.zeros(2), np.zeros(2), np.zeros(2)
    L1[0] = sizeL

    L2[0] = -sizeL / 2
    L3[0] = -sizeL / 2

    return L1, L2, L3

def integrand_generator(L1, L2, L3, cl_phi_interp, lcl_interp, ocl_interp, ellmin, ellmax):
    """
    Closure for capturing fixed L1, L2, L3 etc. and returning the low L approximation of the N2 bias to the lesning bispectrum.
    """

    def integrand_N2(ell):
        
        #print(ell.shape)
        sizeell = vect_modulus(ell)
        ellminusL1 = ell - L1
        sizeellminusL1 = vect_modulus(ellminusL1)
        ellminusL2 = ell - L2
        sizeellminusL2 = vect_modulus(ellminusL2)
        ellminusL3 = ell - L3
        sizeellminusL3 = vect_modulus(ellminusL3)
        ellplusL3 = ell + L3
        sizeellplusL3 = vect_modulus(ellplusL3)
        sizeL1 = vect_modulus(L1)
        sizeL2 = vect_modulus(L2)
        sizeL3 = vect_modulus(L3)

        if sizeell <= ellmax and  sizeellminusL1 <= ellmax and sizeell >= ellmin and  sizeellminusL1 >= ellmin:
            Fint1int = bigF(ell, L1-ell, sizeell, sizeellminusL1, lcl_interp, ocl_interp)
            N2_A = 2/(2*np.pi)**2*Fint1int * cl_phi_interp(sizeL2) * cl_phi_interp(sizeL3) * lcl_interp(sizeell) * dotprod(ell, L2) * dotprod(ell, L3)   
        else:
            N2_A = 0

        if sizeell <= ellmax and sizeellminusL1  <= ellmax and sizeell >= ellmin and  sizeellminusL1 >= ellmin and sizeellplusL3 <= ellmax and sizeellplusL3 >= ellmin:
            Fint1int = bigF(ell, L1-ell, sizeell, sizeellminusL1, lcl_interp, ocl_interp)
            N2_B = - 2/(2*np.pi)**2*Fint1int * cl_phi_interp(sizeL2) * cl_phi_interp(sizeL3) * lcl_interp(sizeellplusL3) * dotprod(ellplusL3, L2) * dotprod(ellplusL3, L3)   
        else:
            N2_B = 0

        return N2_A + N2_B
    return integrand_N2


def compute_integral(L, cl_phi_interp, lcl_interp, ctot_interp, ctotprime_interp, lclprime_interp, lcldoubleprime_interp, ellmin, ellmax):
    """
    Compute the integral directly for the !series expansion! in the low L approximation of the N2 bias to the lensing bispectrum.
    """
    def integrand_N2(ell):
        # Only compute the integrand if ell is within bounds
        if ellmin <= ell <= ellmax:
            integrand = (2 * np.pi) ** 2 * cl_phi_interp(L) ** 2 * 1 / (32 * ctot_interp(ell) ** 3) * L ** 6 * np.pi * (
                -ell * ctotprime_interp(ell) * (8 * lcl_interp(ell) ** 2 + 6 * ell * lcl_interp(ell) * lclprime_interp(ell) + ell ** 2 * lclprime_interp(ell) ** 2) 
                + ctot_interp(ell) * (32 * lcl_interp(ell) ** 2 + 6 * ell ** 2 * lclprime_interp(ell) ** 2 + ell * lcl_interp(ell) * (41 * lclprime_interp(ell) + 3 * ell * lcldoubleprime_interp(ell)))
            )
            return integrand
        else:
            return 0

    # integrate
    result, error = quad(integrand_N2, ellmin, ellmax, limit=500)
    return result, error

def compute_integrand_values(L, cl_phi_interp, lcl_interp, ctot_interp, ctotprime_interp, lclprime_interp, lcldoubleprime_interp, ellmin, ellmax):
    """
    Compute the values of the integrand used in the series expansion above for each ell from ellmin to ellmax for a given L.
    """
    ell_values = np.arange(ellmin, ellmax + 1)  # Ensuring ell values are within the range
    integrand_values = []

    for ell in ell_values:
        integrand = (2 * np.pi) ** 2 * cl_phi_interp(L) ** 2 * 1 / (32 * ctot_interp(ell) ** 3) * L ** 6 * np.pi * (
            -ell * ctotprime_interp(ell) * (8 * lcl_interp(ell) ** 2 + 6 * ell * lcl_interp(ell) * lclprime_interp(ell) + ell ** 2 * lclprime_interp(ell) ** 2) 
            + ctot_interp(ell) * (32 * lcl_interp(ell) ** 2 + 6 * ell ** 2 * lclprime_interp(ell) ** 2 + ell * lcl_interp(ell) * (41 * lclprime_interp(ell) + 3 * ell * lcldoubleprime_interp(ell)))
        )
        #print('int testing', integrand)
        integrand_values.append(integrand)

    return ell_values, integrand_values




############ Main code ##########
integration_limits = [[ellmin, ellmax], [ellmin, ellmax]] #Don't need to use circular limits as the conditions in the integrand function set integrand to zero outside of the circle.
series_integration_limits = [ellmin,ellmax]
bin_edges = np.array([10,20,30,40])#,,50,60,70, 80, 90, 100, 110, 120])
bin_mid = 0.5*(bin_edges[1:] + bin_edges[:-1])


kappa_norm, kappa_curl_norm = {}, {}
kappa_norm['TT'], kappa_curl_norm['TT'] = cs.norm_quad.qtt('lens',lmax,rlmin,rlmax,lcl,ocl,lfac='k')

phi_norm, phi_curl_norm = {}, {}
phi_norm['TT'], phi_curl_norm['TT'] = cs.norm_quad.qtt('lens',lmax,rlmin,rlmax,lcl,ocl,lfac='')

################# Some testing 
l_bins = np.arange(1,100,10) #These will be the different lensing L values we will compute the low L bispectrum for, low L approx valid until approximately L of 100

output_series = []
output_full = []
integral_direct_sum = []

for index, item in enumerate(l_bins):
    L1, L2, L3 = make_equilateral_L(item)
    sizeL1 = vect_modulus(L1)
    norm_factor = kappa_norm['TT'][int(sizeL1)]
    norm_factor_phi = phi_norm['TT'][int(sizeL1)]
    ell_int, integrand = compute_integrand_values(sizeL1, cl_phi_interp, lcl_interp, ctot_interp, ctotprime_interp, lclprime_interp, lcldoubleprime_interp, ellmin, ellmax)
    integral_direct_sum_intermediate = np.sum(integrand) #This computes the direct sum of the integrand for the current L value (set by i). Since the integrand calculated for integer ell spacing this is integral.
    integral_direct_sum.append(norm_factor_phi*integral_direct_sum_intermediate) #Remember to compute the full integral must include the normalisation factor. Recall the form of the integral etc are for phi, not kappa so need norm_factor_phi

    # For each L output the integrand. Note that integral_direct_sum has the value of the integral for each value of L in l_bins stored in the array.
    np.savetxt('integrand_direct_sum_'+str(sizeL1)+'.txt', (ell_int, integrand)) 
    
    # Now compute the low L approximation without the series expansion using vegas as 2D integral


    ######### LOOKS LIKE SENDING KAPPA INSTEAD OF PHI TO THE BELOW!!! 12/7/24 CHECK THIS
    integrand_function = integrand_generator(L1, L2, L3, cl_phi_interp, lcl_interp, ctot_interp, ellmin, ellmax) 
    integrator = vegas.Integrator(integration_limits)
    result = integrator(integrand_function, nitn=15, neval=1000)

    # Now compute the series expansion using the built in quad routine from scipy
    integral_result, integral_error = compute_integral(sizeL1, cl_kappa_interp, lcl_interp, ctot_interp, ctotprime_interp, lclprime_interp, lcldoubleprime_interp, ellmin, ellmax)

    from gvar import mean
    result_mean = mean(result) # Low L approximation, no series expansion.
    print('mean', norm_factor_phi*result_mean) # Low L approx
    print('mean series', norm_factor_phi*integral_result, norm_factor_phi*integral_error) # Low L approx, series expansion, quad integration routine.
    print('direct sum', norm_factor_phi*integral_direct_sum_intermediate) # Low L approx, series expansion, direct sum.
    output_series.append(norm_factor_phi*integral_result)
    output_full.append(norm_factor_phi*result_mean)


# Print some power spectra for checking
np.savetxt('ctotprime.txt', (L,ctotprime))
np.savetxt('lclprime.txt', (L,lclprime))
np.savetxt('lcldoubleprime.txt', (L, lcldoubleprime))
np.savetxt('lcl.txt', (L, lcl))
np.savetxt('ocl.txt', (L, ocl))
np.savetxt('norm_phi', ())

# Print the outputs from low L vegas, Low L series quad, low L series direct sum
np.savetxt('SERIES_lowLN2_out.txt', (l_bins, output_series))
np.savetxt('lowLN2_out.txt', (l_bins, output_full))
np.savetxt('new_direct_sum_integeral.txt', (l_bins, integral_direct_sum))
