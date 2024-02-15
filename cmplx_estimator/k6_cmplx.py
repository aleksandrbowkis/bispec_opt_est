##### This calculated the factored form of the optimal estimator using complex coefficients.
##### Note must use the form of cmblensplus that has been edited such that the bisepctrum estimator can take
##### complex fields on the sphere (it does this by passing the real and imaginary parts seperately
##### USE CONDA ENVIRONMENT cmplx_fld_lensplus
##### See notes in notebook from 6th january 2024 for derivation of the coefficients in this factorised estimator.

import numpy as np
import tqdm
import healpy as hp
import sys, os
sys.path.append('/home/amb257/software/cmplx_cmblensplus/wrap')
sys.path.append('/home/amb257/software/cmplx_cmblensplus/utils')
import matplotlib.pyplot as plt
# from cmblensplus/wrap/
import basic
import curvedsky as cs
#import lenspyx
import cmath

################ Functions ################

def hpalm2lensplus(hpalm):
    sizehpalm = np.shape(hpalm)
    lpsize = int(-0.5 + 0.5*np.sqrt(1+8*sizehpalm[0]))
    lpalm = np.zeros((lpsize,lpsize), dtype = complex)
    indices = np.triu_indices(lpsize)
    lpalm[indices] = hpalm
    lpalm = lpalm.T
    return lpalm

def filter_alms(alm1, alm2, alm3, alm4, octt):
    Fl = np.zeros((rlmax+1,rlmax+1))
    for l in range(rlmin,rlmax+1):
        Fl[l,0:l+1] = 1./octt[l]

    filt_alm1 = alm1 * Fl[:,:]
    filt_alm2 = alm2 * Fl[:,:]
    filt_alm3 = alm3 * Fl[:,:]
    filt_alm4 = alm4 * Fl[:,:]

    return filt_alm1, filt_alm2, filt_alm3, filt_alm4

def avQE(ctt, t1alm, t2alm, lmax, rlmin, rlmax):
    #ctt - lensed temp power spectrum for the sims t1alm, t2alm
    #t1alm, t2alm - the lensed temp alms for each sim
    #lmax - max l of output alm
    #rlmin, rlmax - the min/max l for input alms
    QE1_glm, QE1_clm = {}, {}
    QE1_glm['TT'], QE1_clm['TT'] = cs.rec_lens.qtt(lmax, rlmin, rlmax, ctt, t1alm, t2alm, nside_t = 2048,gtype='k')
    QE2_glm, QE2_clm = {}, {}
    QE2_glm['TT'], QE2_clm['TT'] = cs.rec_lens.qtt(lmax, rlmin, rlmax, ctt, t2alm, t1alm, nside_t = 2048, gtype='k')
    QEav = 0.5 * (QE1_glm['TT'] + QE2_glm['TT'])
    return QEav


################ Parameters ###############

lmax = 2000
Tcmb  = 2.726e6    # CMB temperature
rlmin, rlmax = 2, 3000 # CMB multipole range for reconstruction
nside = 2048
bstype = 'fold'
nsims = 448 # Number of simulations to average over (in sets of 3) 
# Note in submission script nsims = number of command line arguments to pass (any more and start duplicating different datas)

a = complex(0,np.sqrt(2.0))
c = -2.0
d = complex(0, 0.5*(np.sqrt(2)+np.sqrt(10))/np.sqrt(3))
e = complex(0, 0.5*(np.sqrt(2)-np.sqrt(10))/np.sqrt(3))

################ Power spectra ################

ls, cl_unl, cl_len, phi_camb = np.loadtxt('/home/amb257/kappa_bispec/make_sims_parallel/camb_lencl_phi.txt')
L = np.arange(rlmax+1)
Lfac = (L*(L+1.))**2/(2*np.pi)
lcl = cl_len[0:rlmax+1] / Tcmb**2

#Make noise power spectra
theta_fwhm = 1.4 #In arcminutes
sigma_noise = 10 #in muK-arcmin
arcmin2radfactor = np.pi / 60.0 / 180.0
noise_cl = (sigma_noise*arcmin2radfactor/Tcmb)**2*np.exp(L*(L+1.)*(theta_fwhm*arcmin2radfactor)**2/np.log(2.)/8.)
ocl = np.copy(lcl) + noise_cl

################ Main code ###################

#### First read in the simulations. Note these are in healpy form
data_index = int(sys.argv[1])
sim_index = int(sys.argv[2]) #For each data realisation this ranges from 0-447:1

if data_index == sim_index:
    print('cannot use one simulation as both a data and mock simulation')
    sys.exit() #exit the program

# Now sort out which set of 3 simulations we'll use for this evaluation. Note move through simulations cyclicly returning to start once reach the 448th.
all_sims = np.arange(nsims)

where_to_start = np.concatenate((all_sims[sim_index:], all_sims[:sim_index]))

avail_sims = where_to_start[where_to_start != data_index] #Remove the data simulation from the list of available simulations to average over

sim1_index = avail_sims[0] #index 0 is the first simulation
sim2_index = avail_sims[1]
sim3_index = avail_sims[2]

#load in the 'data'
T_data = hp.read_map("/home/amb257/rds/hpc-work/kappa_bispec/simulations/T_phi_noise_"+str(data_index)+".fits")

#load in sim 1 (maps) #testing the new fixed amplitude alm sims. Keep data the same for comparison
T_sim1 = hp.read_map("/home/amb257/rds/hpc-work/kappa_bispec/simulations/T_phi_noise_"+str(sim1_index)+".fits")
T_sim2 = hp.read_map("/home/amb257/rds/hpc-work/kappa_bispec/simulations/T_phi_noise_"+str(sim2_index)+".fits")
T_sim3 = hp.read_map("/home/amb257/rds/hpc-work/kappa_bispec/simulations/T_phi_noise_"+str(sim3_index)+".fits")

#Now convert map to alm
T_data_alm = hp.map2alm(T_data, rlmax)
T_sim1_alm = hp.map2alm(T_sim1, rlmax)
T_sim2_alm = hp.map2alm(T_sim2, rlmax)
T_sim3_alm = hp.map2alm(T_sim3, rlmax)

#convert from hp to lensplus
lp_T_data_alm = hpalm2lensplus(T_data_alm)
lp_T_sim1_alm = hpalm2lensplus(T_sim1_alm)
lp_T_sim2_alm = hpalm2lensplus(T_sim2_alm)
lp_T_sim3_alm = hpalm2lensplus(T_sim3_alm)

#filter by 1/observed power spec
lp_T_data_alm, lp_T_sim1_alm, lp_T_sim2_alm, lp_T_sim3_alm = filter_alms(lp_T_data_alm, lp_T_sim1_alm, lp_T_sim2_alm, lp_T_sim3_alm, ocl)

#Calculate the quadratic estimators
#0 stands for data, 1,2,3 for the sims
kappa_00_glm = avQE(lcl, lp_T_data_alm, lp_T_data_alm, lmax, rlmin, rlmax)
kappa_01_glm = avQE(lcl, lp_T_data_alm, lp_T_sim1_alm, lmax, rlmin, rlmax)
kappa_02_glm = avQE(lcl, lp_T_data_alm, lp_T_sim2_alm, lmax, rlmin, rlmax)
kappa_12_glm = avQE(lcl, lp_T_sim1_alm, lp_T_sim2_alm, lmax, rlmin, rlmax)
kappa_13_glm = avQE(lcl, lp_T_sim1_alm, lp_T_sim3_alm, lmax, rlmin, rlmax)
kappa_23_glm = avQE(lcl, lp_T_sim2_alm, lp_T_sim3_alm, lmax, rlmin, rlmax)

#Compute QE normalisation
phi_norm, phi_curl_norm = {}, {}
phi_norm['TT'], phi_curl_norm['TT'] = cs.norm_quad.qtt('lens',lmax,rlmin,rlmax,lcl,ocl,lfac='k')

#Normalise QE's
kappa_00_glm *= phi_norm['TT'][:,None]
kappa_01_glm *= phi_norm['TT'][:,None]
kappa_02_glm *= phi_norm['TT'][:,None]
kappa_12_glm *= phi_norm['TT'][:,None]
kappa_13_glm *= phi_norm['TT'][:,None]
kappa_23_glm *= phi_norm['TT'][:,None]

#Find the bin edges and norm for bispec estimator (need for all terms)
#because low l bins mix in folded configs which are large and negative so pull the estimate down.
bin_edges = [20,40,60,80,100,200,300,400,500, 600, 700, 800, 900, 1000]
bin_edges = np.array(bin_edges)
nbins = 13 #change this if change the bins above

#bispec_norm = np.loadtxt('bispec_norm.txt')
#bst controls accuracy of calc
bispec_norm = cs.bispec.bispec_norm(nbins,bin_edges, bstype=bstype, bst=4)
bin_mid = 0.5*(bin_edges[1:] + bin_edges[:-1])

#alms for complex field to pass to the bispec estimator. Will pass real part first and then imaginary part
alm_real = kappa_00_glm + c*kappa_12_glm
alm_imag = a.imag*(kappa_01_glm+kappa_02_glm) + d.imag*kappa_13_glm + e.imag*kappa_23_glm

#Compute (unnormalised) bispec for these complex field alms.
bispec_unnorm = cs.bispec.bispec_bin(nbins,bin_edges,lmax,alm_real,alm_imag, bst=4, bstype=bstype)

#Normalise
bispec = bispec_unnorm * np.sqrt(4*np.pi)/bispec_norm

#Save output
np.savetxt("/home/amb257/rds/hpc-work/kappa_bispec/optimal_est/onesimterms/cmplx_multidata/data"+str(data_index)+"/folded_multipledata_cmplx_data"+str(data_index)+"_simstart"+str(sim1_index)+".txt",(bin_mid, bispec))
