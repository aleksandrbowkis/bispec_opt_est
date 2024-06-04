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
print('lcl not interp', lcl[lmax])
ucl = cl_unl[0:rlmax+1] / Tcmb**2 #dimless unlensed T Cl
cl_kappa = Lfac * cl_phi[0:rlmax+1]

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
print('interp lcl',lcl_interp(lmax))
ocl_interp = interp1d(L, ocl, kind='cubic', bounds_error=False, fill_value="extrapolate")

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

def integrand_generator(L1, L2, L3, cl_kappa_interp, lcl_interp, ucl_interp,ocl_interp, ellmin, ellmax):
    """
    Closure for capturing fixed L1, L2, L3 etc. and returning the A type terms in the low L approximation of the N2 bias to the lesning bispectrum.
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
            N2_A = 2/(2*np.pi)**2*Fint1int * cl_kappa_interp(sizeL2) * cl_kappa_interp(sizeL3) * ucl_interp(sizeell) * dotprod(ell, L2) * dotprod(ell, L3)   
        else:
            N2_A = 0

        if sizeell <= ellmax and sizeellminusL1  <= ellmax and sizeell >= ellmin and  sizeellminusL1 >= ellmin and sizeellplusL3 <= ellmax and sizeellplusL3 >= ellmin:
            Fint1int = bigF(ell, L1-ell, sizeell, sizeellminusL1, lcl_interp, ocl_interp)
            N2_B = - 2/(2*np.pi)**2*Fint1int * cl_kappa_interp(sizeL2) * cl_kappa_interp(sizeL3) * ucl_interp(sizeellplusL3) * dotprod(ellplusL3, L2) * dotprod(ellplusL3, L3)   
        else:
            N2_B = 0

        return N2_A + N2_B
       

    return integrand_N2

def integrand_generator_B(L1, L2, L3, cl_kappa_interp, lcl_interp, ucl_interp, ellmin, ellmax):
    """
    Closure for capturing fixed L1, L2, L3 etc. and returning the A type terms in the low L approximation of the N2 bias to the lesning bispectrum.
    """

    def integrand_N2_B(ell):
        
        #print(ell.shape)
        sizeell = vect_modulus(ell)
        ellminusL1 = ell - L1
        sizeellminusL1 = vect_modulus(ellminusL1)
        ellminusL2 = ell - L2
        sizeellminusL2 = vect_modulus(ellminusL2)
        ellplusL3 = ell + L3
        sizeellplusL3 = vect_modulus(ellplusL3)
        sizeL1 = vect_modulus(L1)
        sizeL2 = vect_modulus(L2)
        sizeL3 = vect_modulus(L3)

        if sizeell <= ellmax and sizeellminusL1  <= ellmax and sizeell >= ellmin and  sizeellminusL1 >= ellmin and sizeellplusL3 <= ellmax and sizeellplusL3 >= ellmin:
            Fint1int = bigF(ell, L1-ell, sizeell, sizeellminusL1, lcl_interp, ocl_interp)
            N2_B = - 2*Fint1int * cl_kappa_interp(sizeL2) * cl_kappa_interp(sizeL3) * ucl_interp(sizeellplusL3) * dotprod(ellplusL3, L2) * dotprod(ellplusL3, L3)   
        else:
            N2_B = 0

        return N2_B 

    return integrand_N2_B



############ Main code ##########

integration_limits = [[ellmin, ellmax], [ellmin, ellmax]] #Don't need to use circular limits as the conditions in the integrand function set integrand to zero outside of the circle.

bin_edges = np.array([10,20,30,40])#,,50,60,70, 80, 90, 100, 110, 120])
bin_mid = 0.5*(bin_edges[1:] + bin_edges[:-1])


kappa_norm, kappa_curl_norm = {}, {}
kappa_norm['TT'], kappa_curl_norm['TT'] = cs.norm_quad.qtt('lens',lmax,rlmin,rlmax,lcl,ocl,lfac='k')



print('done')

################# Some testing of resopnse function
#first test dot prod and vect_modulus
l_bins = np.arange(0,100,10)
print(l_bins)
output = []
for i in l_bins:
    L1, L2, L3 = make_equilateral_L(i)
    sizeL1 = vect_modulus(L1)

    integrand_function = integrand_generator(L1,  L2, L3, cl_kappa_interp, lcl_interp, ucl_interp, ocl_interp, ellmin, ellmax)
    integrator = vegas.Integrator(integration_limits)
    result = integrator(integrand_function, nitn=15, neval=1000)

    from gvar import mean
    norm_factor =  kappa_norm['TT'][int(sizeL1)]
    print(result)
    result_mean = mean(result)
    print('mean', norm_factor*result_mean)
    print('stuff', cl_kappa_interp(10) / kappa_norm['TT'][int(10)])
    print('clk', cl_kappa_interp(10))
    print('magnitude', cl_kappa_interp(10)**3 * sizeL1**6 * ellmax**2)
    print('magnitude N0',  sizeL1**6 * ellmax**2 *norm_factor**3)
    print('sizes test',sizeL1, ellmax)
    print((result_mean)*norm_factor)
    output.append(result_mean * norm_factor)

print(output) 
np.savetxt('lowLN2_out.txt', (l_bins, output))