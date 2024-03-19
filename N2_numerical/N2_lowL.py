#### Import modules
import numpy as np
import camb
import vegas 
from scipy.interpolate import interp1d

################ Parameters ###############

lmax = 2000
Tcmb  = 2.726e6    # CMB temperature
rlmin, rlmax = 2, 3000 # CMB multipole range for reconstruction
nside = 2048
bstype = 'fold'
nsims = 448 # Number of simulations to average over (in sets of 3) 
ellmin = 2 
ellmax = 1000

################ Power spectra ################

ls, cl_unl, cl_len, cl_phi = np.loadtxt('/home/amb257/kappa_bispec/make_sims_parallel/camb_lencl_phi.txt')
L = np.arange(rlmax+1)
Lfac = (L*(L+1.) / 2 )**2
lcl = cl_len[0:rlmax+1] / Tcmb**2
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

def integrand_generator(L1, L2, L3, cl_kappa_interp, lcl_interp, ucl_interp, ellmin, ellmax):
    """
    Closure for capturing fixed L1, L2, L3 etc. and returning a 2D integrand function.
    """

    def integrand_N2_A(ell):
        
        #print(ell.shape)
        ell_size = vect_modulus(ell)
        ellminusL1 = ell - L1
        sizeellminusL1 = vect_modulus(ellminusL1)
        
        sizeL2 = vect_modulus(L2)
        sizeL3 = vect_modulus(L3)
                                                                
        
        if ell_size <= ellmax and  sizeellminusL1 <= ellmax and ell_size >= ellmin and  sizeellminusL1 >= ellmin:
            Fint1int = bigF(ell, L1-ell, ell_size, sizeellminusL1, lcl_interp, ocl_interp)
            N2_A =  Fint1int * cl_kappa_interp(sizeL2) * cl_kappa_interp(sizeL3) * ucl_interp(sizeellminusL1) * dotprod(ellminusL1, L2) * dotprod(ellminusL1, L3)   
        else:
            N2_A = 0

        return N2_A 

    return integrand_N2_A


############ Main code ##########

integration_limits = [[ellmin, ellmax], [ellmin, ellmax]] #Don't need to use circular limits as the conditions in the integrand function set integrand to zero outside of the circle.

bin_edges = np.array([20,40,60,80,100,200,300,400,500, 600, 700, 800, 900, 1000])
bin_mid = 0.5*(bin_edges[1:] + bin_edges[:-1])

for i in bin_mid:
    L1, L2, L3 = make_fold_L(i)
    integrand_function = integrand_generator(L1,  L2, L3, cl_kappa_interp, lcl_interp, ucl_interp, ellmin, ellmax)
    integrator = vegas.Integrator(integration_limits)
    result = integrator(integrand_function, nitn=10, neval=1000)

    #Now normalise
    norm_factor = (2*np.pi)^(-2)
    print(i, result)
print('done')