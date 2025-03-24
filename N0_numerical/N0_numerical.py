import numpy as np
import vegas
from scipy.integrate import quad
from scipy.interpolate import interp1d
import sys, os
sys.path.append('/home/amb257/software/cmplx_cmblensplus/wrap')
sys.path.append('/home/amb257/software/cmplx_cmblensplus/utils')
# from cmblensplus/wrap/
import curvedsky as cs
import flatsky as fs
import multiprocessing as mp


# Parameters
lmax = 1000
Tcmb  = 2.726e6    # CMB temperature in microkelvin?
bstype = 'equi'
ellmin, ellmax = 2, 3000
rlmin, rlmax = 2, 3000 # CMB multipole range for reconstruction

# Define the directory where power spec are stored
input_dir = "../Power_spectra"

# Load the power spec
L = np.arange(0,3000+1,1)
ucl = np.loadtxt(os.path.join(input_dir, "unlensed_clTT_lmax8000.txt"))
gcl = np.loadtxt(os.path.join(input_dir, "glensed_clTT_lmax8000.txt"))
#ctot = np.loadtxt(os.path.join(input_dir, "lensed_clTT_lmax8000.txt")) 
lcl = np.loadtxt(os.path.join(input_dir, "lensed_clTT_lmax8000.txt"))
ucl = ucl[0:3001]
gcl = gcl[0:3001]
lcl = lcl[0:3001]
# ctot = ctot[0:2001]

#Make noise power spectra
theta_fwhm = 1.4 #In arcminutes
sigma_noise = 10 #in muK-arcmin
arcmin2radfactor = np.pi / 60.0 / 180.0
noise_cl = (sigma_noise*arcmin2radfactor/Tcmb)**2*np.exp(L*(L+1.)*(theta_fwhm*arcmin2radfactor)**2/np.log(2.)/8.)
ctot = np.copy(lcl) + noise_cl

# Now interpolate them
#cl_phi_interp = interp1d(L, cl_phi, kind='cubic', bounds_error=False, fill_value="extrapolate")
ucl_interp = interp1d(L, ucl, kind='cubic', bounds_error=False, fill_value="extrapolate")
lcl_interp = interp1d(L, lcl, kind='cubic', bounds_error=False, fill_value="extrapolate")
gcl_interp = interp1d(L, gcl, kind='cubic', bounds_error=False, fill_value="extrapolate")
ctot_interp = interp1d(L, ctot, kind='cubic', bounds_error=False, fill_value="extrapolate")

# Load the normalisation
flat_sky_norm = np.load("/home/amb257/kappa_bispec/bispec_opt_est/N0_numerical/normalisation/flat_sky_norm.npy")
print('shape', np.shape(flat_sky_norm))
flat_sky_norm_interp = interp1d(L, flat_sky_norm, kind='cubic', bounds_error=False, fill_value="extrapolate")
################# Functions ##################

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

# Integrand function use vegas batchintegrand to evaluate integral at multiple points
@vegas.batchintegrand
def integrand_N0_batched(x, L1, L3, gcl_interp, ctot_interp, lcl_interp, ellmin, ellmax):
    x = np.atleast_2d(x) # Make x into a 2d array as expected in code below
    # x[:, 0] is the x-component of ell, x[:, 1] is the y-component of ell
    ell = np.stack((x[:, 0], x[:, 1]), axis=-1)

    sizeell = np.sqrt(np.sum(ell**2, axis=1)) # Allows modulus for each point in the 2D array of points to be computed (N rows of points with 2 columns, for x and y coordinates of a single point)
    L1minusell = L1[np.newaxis, :] - ell # Broadcasting allows every point that vegas sends in batch mode from the same L1
    sizeL1minusell = np.sqrt(np.sum(L1minusell**2, axis=1))
    ellplusL3 = ell + L3[np.newaxis, :]
    sizeellplusL3 = np.sqrt(np.sum(ellplusL3**2, axis=1))

    # mask out disallowed regions
    mask = (ellmin <= sizeell) & (sizeell <= ellmax) & \
           (ellmin <= sizeL1minusell) & (sizeL1minusell <= ellmax) & \
           (ellmin <= sizeellplusL3) & (sizeellplusL3 <= ellmax)

    integrand = np.zeros_like(sizeell)
    
    valid_idx = np.where(mask)  # Only compute valid integrands
    F1 = bigF(ell[valid_idx], L1minusell[valid_idx], sizeell[valid_idx], sizeL1minusell[valid_idx], gcl_interp, ctot_interp)
    F13 = bigF(L1minusell[valid_idx], ellplusL3[valid_idx], sizeL1minusell[valid_idx], sizeellplusL3[valid_idx], gcl_interp, ctot_interp)
    F3 = bigF(-ell[valid_idx], ellplusL3[valid_idx], sizeell[valid_idx], sizeellplusL3[valid_idx], gcl_interp, ctot_interp)
    
    integrand[valid_idx] = (1 / (2*np.pi)**2 * 8 * F1 * F13 * F3 * lcl_interp(sizeell[valid_idx]) * lcl_interp(sizeL1minusell[valid_idx]) * lcl_interp(sizeellplusL3[valid_idx]))
    
    integrand = np.atleast_1d(integrand)

    return integrand

# Function to calculate the result for each L
def compute_for_L(lensingL, gcl_interp, ctot_interp, lcl_interp, ellmin, ellmax, phi_norm):
    L1, L2, L3 = make_equilateral_L(lensingL)
    integration_limits = [[-ellmax, ellmax], [-ellmax, ellmax]]
    integrator = vegas.Integrator(integration_limits)
    
    result = integrator(lambda x: integrand_N0_batched(x, L1, L3, gcl_interp, ctot_interp, lcl_interp, ellmin, ellmax), nitn=12, neval=15000)
    from gvar import mean
    result_mean = mean(result)
    
    # Apply normalization
    norm_factor_phi = phi_norm(lensingL)
    result_mean *= norm_factor_phi**3
    if isinstance(result_mean, np.ndarray):
        result_mean = result_mean[0]
    return result_mean.item()

################ Main code ####################
def main():
    samples = 100 # or fully sampled: int(rlmax-rlmin+1)
    lensingLarray = np.linspace(ellmin, ellmax, samples)
    output_dir = "N0_numerical_BATCHvegas"
    os.makedirs(output_dir, exist_ok=True)

    # Use multiprocessing to parallelise over the lensing L's (calculate integral for each lensing L) 
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(compute_for_L, [(lensingL, gcl_interp, ctot_interp,lcl_interp, ellmin, ellmax, flat_sky_norm_interp) for lensingL in lensingLarray])


    # Convert the results to a numpy array and save
    output = np.array(results)
    np.save(os.path.join(output_dir, "L_N0_num.npy"), lensingLarray)
    np.save(os.path.join(output_dir, "N0_numerical_equi.npy"), output)
    print()

if __name__ == '__main__':
    main()