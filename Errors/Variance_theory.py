
## Program to compute variance in the binned bispectrum based on "biases to primordial non gaussianity measurements from CMB secondary anisotropies"
## Updated to just check whether l1 is an allowed value of the w3j function and sum over that rather than calculating each case of w3j in/out of bin
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('/home/amb257/software/cmplx_cmblensplus/wrap')
sys.path.append('/home/amb257/software/cmplx_cmblensplus/utils')
# from cmblensplus/wrap/
import basic
import curvedsky as cs
import camb

############## Functions #################

#Function to compute the estimator normalisation directly for testing purposes
def Nijk(bin_edges, size_bin_edges):
    N = np.zeros(size_bin_edges-1)
    sum = 0
    for index, item in enumerate(bin_edges[0:size_bin_edges-1]):
        sum = 0
        lower_bound_bin = int(item)
        upper_bound_bin = int(bin_edges[index+1])
        for l3 in range(int(lower_bound_bin/changebins), int(upper_bound_bin/changebins)):
            for l2 in range(int(lower_bound_bin/changebins), int(upper_bound_bin/changebins)):
                #First calculate the l bounds of w3j function (allowed l1 values given l2,3)
                lower_bound_w3j = np.abs(l3 - l2)
                upper_bound_w3j = l3 + l2
                #Calculate the w3j's
                w3j = basic.wigner_funcs.wigner_3j(l3,l2,0,0)
                for l1 in range(lower_bound_bin, upper_bound_bin):
                    if l1 >= lower_bound_w3j and l1 <= upper_bound_w3j:
                        position_l1_in_w3j = l1 - lower_bound_w3j #this is the position of the current value of l1 in the w3j array
                        sum += (2*l1+1)*(2*l2+1)*(2*l3+1) * w3j[position_l1_in_w3j]**2 / (4*np.pi)
        N[index] = sum
    return N
################ Parameters ###############

lmax = 1000
Tcmb  = 2.726e6    # CMB temperature
rlmin, rlmax = 2, 3000 # CMB multipole range for reconstruction
nside = 2048
bstype = 'equi'
bin_edges = [20,40,60,80,100,200,300,400,500, 600, 700, 800, 900, 1000]
bin_edges = np.array(bin_edges)
nbins = 13
size_bin_edges = np.shape(bin_edges)[0]

# Now set the number of non zero permutations which follows from no. of distinct bins used in the estimator (diff for folded and equi)
# and set changebins which changes the bins used st is l/2, l/2, l for folded and l, l, l for equilateral
if bstype == 'fold':
    perms = 2
    changebins =2
elif bstype == 'equi':
    perms = 6
    changebins = 1
else:
    print('neither folded or equilateral configuration, setting to zero')
    perms = 0
    changebins=1

################ Power spectra ################

ls, cl_unl, cl_len, cl_phi = np.loadtxt('/home/amb257/kappa_bispec/make_sims_parallel/camb_lencl_phi.txt')
L = np.arange(0,rlmax+1)
Lfac = (L*(L+1.))**2/(4)
lcl = cl_len[0:rlmax+1] / Tcmb**2
cl_kappa = cl_phi[2:lmax+1] * Lfac[2: lmax+1]
#Now prepend this cl_kappa array with two zero values which take the place of the l=0 and l=1 multipoles. 
# Create an array of two zeros
zeros = np.zeros(2)
# Concatenate the zeros array with cl_kappa
cl_kappa = np.concatenate((zeros, cl_kappa))
#Now cl_kappa starts from l=0 but only has info from l=2. This means can use l1, etc to index positions in the cl_var array defined below.

#Make noise power spectra
theta_fwhm = 1.4 #In arcminutes
sigma_noise = 10 #in muK-arcmin
arcmin2radfactor = np.pi / 60.0 / 180.0
noise_cl = (sigma_noise*arcmin2radfactor/Tcmb)**2*np.exp(L*(L+1.)*(theta_fwhm*arcmin2radfactor)**2/np.log(2.)/8.)
ocl = np.copy(lcl) + noise_cl

#Calculate the quadratic estimator normalisation. This is the N0 bias term to the power spectrum. We want to use the lensing potential power spectrum + N0 bias as the Cl in this variance calculation.
norm_phi, curl_norm = cs.norm_quad.qtt('lens',lmax,rlmin,rlmax,lcl,ocl, lfac='k')

#Create the power spec we want to use in the variance calculation
cl_var = cl_kappa + norm_phi

############### Main code #####################

# Find the bin edges and norm for bispec estimator (need for all terms)
#because low l bins mix in folded configs which are large and negative so pull the estimate down.


N_test = Nijk(bin_edges, size_bin_edges)
bincenters = (bin_edges[1:]+bin_edges[:-1])*.5
#Compute bs normalisation
bispec_norm = cs.bispec.bispec_norm(nbins,bin_edges, bstype=bstype, bst=4)
full_bs_norm = np.sqrt(4*np.pi)/bispec_norm
N = 1 / full_bs_norm

print('n test', N_test)
print('N', N)
#initialise sum, l
l = 0
sum = 0

var = np.zeros(size_bin_edges-1) #Make array to hold the variance in the bispec estimate for each bin
normalising_factor = perms / (4 * np.pi * N_test**2) #Calc norm for the binned bispec. This has the same shape as var - one element for each bin - the norm at that bin.

for index, item in enumerate(bin_edges[0:size_bin_edges-1]):
    lower_bound_bin = int(item)
    upper_bound_bin = int(bin_edges[index+1])
    for l3 in range(int(lower_bound_bin/changebins), int(upper_bound_bin/changebins)):
        for l2 in range(int(lower_bound_bin/changebins), int(upper_bound_bin/changebins)):

            #First calculate the l bounds of w3j function (allowed l1 values given l2,3)
            lower_bound_w3j = np.abs(l3 - l2)
            upper_bound_w3j = l3 + l2
            #Calculate the w3j's
            w3j = basic.wigner_funcs.wigner_3j(l3,l2,0,0)

            for l1 in range(lower_bound_bin, upper_bound_bin):
                if l1 >= lower_bound_w3j and l1 <= upper_bound_w3j:
                    position_l1_in_w3j = l1 - lower_bound_w3j #this is the position of the current value of l1 in the w3j array
                    sum_term = (2*l1+1)*(2*l2+1)*(2*l3+1) * cl_var[l1]* cl_var[l2]*cl_var[l3] * w3j[position_l1_in_w3j]**2
                    sum += sum_term

    var[index] = sum * normalising_factor[index]
    sum = 0 #reset sum to zero so can calculate the variance for the next bin.

#Save results
print('std_new', np.sqrt(var))
np.savetxt("new"+str(bstype)+"_var.txt", (bincenters, var))

