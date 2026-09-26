#!/usr/bin/env python
# coding: utf-8

#Make a guassian map using lensed power spectra. Will take this as data for checking the optimal k6 estimator.

import os
import sys
import matplotlib.pyplot as plt
import healpy as hp, numpy as np

# parameters which impact of the accuracy of the result (and the execution time):

lmax = 4096  # desired lmax of the lensed field. Why power of 2?
dlmax = 1024  # lmax of the unlensed fields is lmax + dlmax.  (some buffer is required for accurate lensing at lmax)
nside = 2048 # The lensed tlm's are computed with healpy map2alm from a lensed map at resolution 'nside_lens'
facres = -1 # The lensed map of resolution is interpolated from a default high-res grid with about 0.7 amin-resolution
            # The resolution is changed by 2 ** facres is this is set.
nsims = 1 # The number of sets of simulations
Tcmb  = 2.726e6    # CMB temperature
#from lenspyx.utils import camb_clfile

i = int(sys.argv[1])

cl_len, _, _, dimless_phi_cl = np.loadtxt('camb_lencl_phi.txt')
lcl = cl_len / Tcmb**2
ell = np.arange(lmax+dlmax)

#Make noise power spectra
theta_fwhm = 1.4 #In arcminutes
sigma_noise = 10 #in muK-arcmin
arcmin2radfactor = np.pi / 60.0 / 180.0
noise_cl = (sigma_noise*arcmin2radfactor/Tcmb)**2*np.exp(ell*(ell+1.)*(theta_fwhm*arcmin2radfactor)**2/np.log(2.)/8.)

#Make alm's for an lensed T using lcl
T_lcl_map = hp.synfast(lcl,nside, lmax=lmax + dlmax, new=True)

#Make noise map
noise_map = hp.synfast(noise_cl, nside, lmax=lmax + dlmax, new=True)

#Make total map
T_len_gauss_noise = T_lcl_map + noise_map

#Save map
hp.write_map("/home/amb257/rds/hpc-work/kappa_bispec/simulations/opt_est_test/T_len_gauss_noise_"+str(i)+".fits", T_len_gauss_noise, overwrite=True)

