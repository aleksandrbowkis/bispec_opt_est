"""Code to compute the normalisation of the quadratic estimator in the flat sky approximation.
    Remains to check if should be using gradient lensed power spectra."""

import numpy as np
import vegas as vg

###### Parameters ####

lmax = 2000
Tcmb  = 2.726e6    # CMB temperature in microkelvin?
bstype = 'equi'
ellmin, ellmax = 2, 2000
rlmin, rlmax = 2, 2000 # CMB multipole range for reconstruction

###### Read in and interpolate power spectra ####
input_dir = "../Power_spectra"

# Load the power spec
L = np.arange(0,2000+1,1)
gcl = np.loadtxt(os.path.join(input_dir, "glensed_clTT_lmax8000.txt"))
ctot = np.loadtxt(os.path.join(input_dir, "lensed_clTT_lmax8000.txt")) # Using nonoise lensed power spectra atm cf Alba's numerical results. Change to ctot inc noise later.
gcl = gcl[0:2001]
ctot = ctot[0:2001]
# Now interpolate them
gcl_interp = interp1d(L, gcl, kind='cubic', bounds_error=False, fill_value="extrapolate")
ctot_interp = interp1d(L, ctot, kind='cubic', bounds_error=False, fill_value="extrapolate")

##### Functions ###
def dotprod(l1, l2):
    # Computes the dot product of two sets of vectors. Note now vectorised so can compute dotproduct of sets of vectors rather than two single vectors.
    return np.sum(l1 * l2, axis=1)

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

@vegas.batchintegrand
def integrand(x, L gcl_interp, ctot_interp):
    x = np.atleast_2d(x) # Make x into a 2d array as expected in code below
    # x[:, 0] is the x-component of ell, x[:, 1] is the y-component of ell
    ell = np.stack((x[:, 0], x[:, 1]), axis=-1)
    sizeell = np.sqrt(np.sum(ell**2, axis=1)) # Allows modulus for each point in the 2D array of points to be computed (N rows of points with 2 columns, for x and y coordinates of a single point)
    Lminusell = L[np.newaxis, :] - ell # Broadcasting allows every point that vegas sends in batch mode from the same L1
    sizeLminusell = np.sqrt(np.sum(Lminusell**2, axis=1))

    # mask out disallowed regions
    mask = (ellmin <= sizeell) & (sizeell <= ellmax) & \
           (ellmin <= sizeLminusell) & (sizeLminusell <= ellmax)
    
    integrand = np.zeros_like(sizeell)

    valid_idx = np.where(mask)  # Only compute valid integrands
    littlef1 = response_func(gcl_interp, ell[valid_idx], Lminusell[valid_idx], sizeell[valid_idx], sizeLminusell[valid_idx])
    F1 = bigF(ell[valid_idx], Lminusell[valid_idx], sizeell[valid_idx], sizeLminusell[valid_idx], gcl_interp, ctot_interp)

    integrand[valid_idx] = 1 / (2*np.pi)**2 * F1 * littlef1
    integrand = np.atleast_1d(integrand)
    return integrand

def compute_normalisation(L, gcl_interp, ctot_interp, ellmin, ellmax):
    integration_limits = [[ellmin, ellmax], [ellmin, ellmax]]
    integrator = vegas.Integrator(integration_limits)
    
    result = integrator(lambda x: integrand(x, L, gcl_interp, ctot_interp, , ellmin, ellmax), nitn=1, neval=1000)

    from gvar import mean
    result_mean = mean(result)

    norm = 1 / result_mean
    return norm

##### Main #####

