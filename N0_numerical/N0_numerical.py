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

################ Parameters ###############

lmax = 1000
Tcmb  = 2.726e6    # CMB temperature in microkelvin?
rlmin, rlmax = 2, 3000 # CMB multipole range for reconstruction
nside = 2048
bstype = 'fold'
nsims = 448 # Number of simulations to average over (in sets of 3) 
ellmin = 2 
ellmax = 3000 ##### check!!!! vs sims

################ Power spectra ################

ls, cl_unl, cl_len, cl_phi = np.loadtxt('/home/amb257/kappa_bispec/make_sims_parallel/camb_lencl_phi.txt')
L = np.arange(rlmax+1)
Lfac = (L*(L+1.) / 2 )**2
lcl = cl_len[0:rlmax+1] / Tcmb**2
print('lcl not interp', lcl[1000])
ucl = cl_unl[0:rlmax+1] / Tcmb**2 #dimless unlensed T Cl
cl_kappa = Lfac * cl_phi[0:3001]

#Make noise power spectra
theta_fwhm = 1.4 #In arcminutes
sigma_noise = 10 #in muK-arcmin
arcmin2radfactor = np.pi / 60.0 / 180.0
noise_cl = (sigma_noise*arcmin2radfactor/Tcmb)**2*np.exp(L*(L+1.)*(theta_fwhm*arcmin2radfactor)**2/np.log(2.)/8.)
ocl = np.copy(lcl) + noise_cl

# Interpolation functions for cl_kappa and ucl
cl_kappa_interp = interp1d(L, cl_kappa, kind='cubic', bounds_error=False, fill_value="extrapolate")
ucl_interp = interp1d(L, ucl, kind='cubic', bounds_error=False, fill_value="extrapolate")
lcl_interp = interp1d(L, lcl, kind='cubic', bounds_error=False, fill_value="extrapolate")
print('interp lcl',lcl_interp(1000))
ocl_interp = interp1d(L, ocl, kind='cubic', bounds_error=False, fill_value="extrapolate")

################# Functions ##################

def dotprod(l1, l2):
    # This allows dot product calculations between two vectors or batches of vectors
    return np.sum(l1 * l2, axis=-1)


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

######### Main Code ########

def norm_generator(L):
    """
    Closure for calculating normalising factor for eqn (3.12) in Bispectrum paper
    """
    def norm_integrand(ell):
        Lminusell = L - ell
        sizeLminusell = vect_modulus(Lminusell)
        sizeell = vect_modulus(ell)

        valid = (sizeell <= ellmax) & (sizeLminusell <= ellmax) & (sizeell >= ellmin) & (sizeLminusell >= ellmin)
        results = np.zeros(ell.shape[0])

        if np.any(valid):
            valid_ell = ell[valid]
            valid_sizeell = sizeell[valid]
            valid_sizeLminusell = sizeLminusell[valid]

            FintLint = bigF(valid_ell, L - valid_ell, valid_sizeell, valid_sizeLminusell, lcl_interp, ocl_interp)
            lowerfintLint = response_func(lcl_interp, valid_ell, L - valid_ell, valid_sizeell, valid_sizeLminusell)
            results[valid] = 1/(2*np.pi) * FintLint * lowerfintLint
        return results
    return norm_integrand

def integrand_generator(L1, L3, ocl_interp, ellmin, ellmax):
    """
    Closure for capturing fixed L1, L2, L3 etc. and returning the A type terms in the low L approximation of the N2 bias to the lensing bispectrum.
    """

    def integrand_N1(ell):
        sizeell = vect_modulus(ell)
        ellminusL1 = ell - L1
        sizeellminusL1 = vect_modulus(ellminusL1)
        ellplusL3 = ell + L3
        sizeellplusL3 = vect_modulus(ellplusL3)
        
        valid = (sizeell <= ellmax) & (sizeellminusL1 <= ellmax) & (sizeell >= ellmin) & (sizeellminusL1 >= ellmin)
        results = np.zeros(ell.shape[0])  # Ensure output shape is correct

        if np.any(valid):
            valid_ell = ell[valid]
            valid_sizeell = sizeell[valid]
            valid_sizeellminusL1 = sizeellminusL1[valid]
            valid_sizeellplusL3 = sizeellplusL3[valid]
            Fint1int = bigF(valid_ell, L1 - valid_ell, valid_sizeell, valid_sizeellminusL1, lcl_interp, ocl_interp)
            F1int3int = bigF(L1 - valid_ell, L3 + valid_ell, valid_sizeellminusL1, valid_sizeellplusL3, lcl_interp, ocl_interp)
            Fint3int = bigF(-valid_ell, L3 + valid_ell, valid_sizeell, valid_sizeellplusL3, lcl_interp, ocl_interp)
            results[valid] = 8 / (2 * np.pi)**2 * ocl_interp(valid_sizeell) * ocl_interp(valid_sizeellminusL1) * ocl_interp(valid_sizeellplusL3) * Fint1int * F1int3int * Fint3int
        return results
    return integrand_N1


########## Main ########

integration_limits = [[ellmin, ellmax], [ellmin, ellmax]]

bin_edges = np.array([0, 200, 400, 600, 800, 1000])
bin_mid = 0.5*(bin_edges[1:] + bin_edges[:-1])

kappa_norm, kappa_curl_norm = {}, {}
kappa_norm['TT'], kappa_curl_norm['TT'] = cs.norm_quad.qtt('lens',lmax,rlmin,rlmax,lcl,ocl,lfac='k')
print('kappa_norm', kappa_norm['TT'][150])

# Calculate normalisation (N.B. for equi all norms are the same. for folded they won't be).
for i in bin_mid:
    L1, L2, L3 = make_equilateral_L(i)
    sizeL1 = vect_modulus(L1)
    sizeL2 = np.int(vect_modulus(L2))
    sizeL3 = np.int(vect_modulus(L3))
    norm_integrand_function = norm_generator(L1)
    integrator = vegas.Integrator(integration_limits)
    norm_integral1 = integrator(norm_integrand_function, nitn=10, neval=1000)
    norm1 = sizeL1**2 / (2 * norm_integral1)
    norm_integral2 = integrator(norm_integrand_function, nitn=10, neval=1000)
    norm2 = sizeL2**2 / (2 * norm_integral2)
    norm_integral3 = integrator(norm_integrand_function, nitn=10, neval=1000)
    norm3 = sizeL3**2 / (2 * norm_integral3)
    print('norm',norm1, norm2, norm3)

N0_final = np.zeros(np.shape(bin_mid)[0])
                    
for index, item in enumerate(bin_mid):
    L1, L2, L3 = make_equilateral_L(item)
    sizeL1 = int(vect_modulus(L1))
    sizeL2 = int(vect_modulus(L2))
    sizeL3 = int(vect_modulus(L3))
    print(sizeL1)
    integrand_function = integrand_generator(L1, L3, ocl_interp, ellmin, ellmax)
    
    output = integrator(integrand_function, nitn=10, neval=1000)
    print(np.shape(output))
    #N0_final[index] = norm1 * norm2 * norm3 * output[0]


np.savetxt('N0_numerical', (bin_mid, N0_final))