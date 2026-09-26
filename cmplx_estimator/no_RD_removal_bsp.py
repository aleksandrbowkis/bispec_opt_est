##### This calculates the bispectrum from simulations without doing any RD bispectrum removal.

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

def filter_alms(alm1, octt):
    Fl = np.zeros((rlmax+1,rlmax+1))
    for l in range(rlmin,rlmax+1):
        Fl[l,0:l+1] = 1./octt[l]

    filt_alm1 = alm1 * Fl[:,:]

    return filt_alm1

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
bstype = 'equi'
# Note in submission script nsims = number of command line arguments to pass (any more and start duplicating different datas)

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
sim1_index = int(sys.argv[1])

#load in sim 1 (maps) #testing the new fixed amplitude alm sims. Keep data the same for comparison
# T_sim1 = hp.read_map("/home/amb257/rds/hpc-work/kappa_bispec/simulations/T_phi_noise_"+str(sim1_index)+".fits")
T_sim1 = hp.read_map("/home/amb257/rds/rds-dirac-dp002/amb257/paired_sims_ch2/Tlen_noise_"+str(sim1_index)+".fits")

#Now convert map to alm
T_sim1_alm = hp.map2alm(T_sim1, rlmax)

#convert from hp to lensplus
lp_T_sim1_alm = hpalm2lensplus(T_sim1_alm)

#filter by 1/observed power spec
lp_T_sim1_alm= filter_alms(lp_T_sim1_alm, ocl)

#Calculate the quadratic estimators
#0 stands for data, 1,2,3 for the sims
kappa_glm = avQE(lcl, lp_T_sim1_alm, lp_T_sim1_alm, lmax, rlmin, rlmax)

#Compute QE normalisation
phi_norm, phi_curl_norm = {}, {}
phi_norm['TT'], phi_curl_norm['TT'] = cs.norm_quad.qtt('lens',lmax,rlmin,rlmax,lcl,ocl,lfac='k')

#Normalise QE's
kappa_glm *= phi_norm['TT'][:,None]

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
alm_real = kappa_glm
alm_imag = np.zeros_like(kappa_glm)

#Compute (unnormalised) bispec for these complex field alms.
bispec_unnorm = cs.bispec.bispec_bin(nbins,bin_edges,lmax,alm_real,alm_imag, bst=4, bstype=bstype)

#Normalise
bispec = bispec_unnorm * np.sqrt(4*np.pi)/bispec_norm

#Save output
np.savetxt("/home/amb257/rds/hpc-work/kappa_bispec/bias_from_sims/no_RD_bispec/equi/ch2sims_"+str(sim1_index)+".txt",(bin_mid, bispec))
