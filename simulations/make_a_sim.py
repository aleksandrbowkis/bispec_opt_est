#!/usr/bin/env python
# coding: utf-8

#Make sets of simulations I, i, i' where
#I is an unlensed CMB T1, lensed by phi1 then noise added
#i is the same unlensed CMB T1, lensed by phi1 WITHOUT noise
#i' is a new unlensed CMB T2, lensed by the same phi1
#Then we make another set of three sims with different unlensed T and different phi.
import lenspyx
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
from lenspyx.utils import camb_clfile


_, cl_unl, _, dimless_phi_cl = np.loadtxt('camb_lencl_phi.txt')
ucl = cl_unl / Tcmb**2
ell = np.arange(lmax+dlmax)

# #Read in power spectra. Make dimensionless.
# cls_path = os.path.join(os.path.dirname(os.path.abspath(lenspyx.__file__)), 'data', 'cls')
# cl_unl = camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
# ell = np.arange(lmax+dlmax)
# dimless_phi_cl = cl_unl['pp']
# ucl = cl_unl['tt'] / Tcmb**2

#Make noise power spectra
theta_fwhm = 1.4 #In arcminutes
sigma_noise = 10 #in muK-arcmin
arcmin2radfactor = np.pi / 60.0 / 180.0
noise_cl = (sigma_noise*arcmin2radfactor/Tcmb)**2*np.exp(ell*(ell+1.)*(theta_fwhm*arcmin2radfactor)**2/np.log(2.)/8.)

#Make simulations required for one realisation of N1 (except the 2pt disconnected piece which will be taken from the next set of sims up)

#read in command line arg (n.b. the 0 element of the array is the program name).
# This is the number of the start of this set of 4 sims.
iterator = int(sys.argv[1])
#Make alm's for an unlensed T using ucl
T_alm_unl = hp.synalm(ucl, lmax=lmax + dlmax, new=True)
#Make alm's for phi
phi_alm = hp.synalm(dimless_phi_cl, lmax=lmax + dlmax, new=True)
#Transform phi alm's into a deflection field. Multiply alm's by sqrt[l(l+1)]
deflec_alm = hp.almxfl(phi_alm, np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2)))
#Make noise map
noise_map = hp.synfast(noise_cl, nside, lmax=lmax + dlmax, new=True)

#Make simulation i and I
#Start by deflecting the unlensed T by the deflection field. This is simulation i.
first_lens_T_no_noise = lenspyx.alm2lenmap(T_alm_unl, [deflec_alm, None], nside, facres=facres)
hp.write_map("/home/amb257/rds/hpc-work/kappa_bispec/simulations/T_phi_"+str(iterator)+".fits", first_lens_T_no_noise, overwrite=True)
#Add the noise map to the lensed T map
first_lens_T_noise = first_lens_T_no_noise + noise_map
hp.write_map("/home/amb257/rds/hpc-work/kappa_bispec/simulations/T_phi_noise_"+str(iterator)+".fits", first_lens_T_noise, overwrite=True)

#Make simulation Tprimephi (different unlensed T by lensed by the same phi)
#Note just doing alm2lenmap again just spits out the same thing rather than a new one.
#Try making a new T_alm_unl then repeating? has to be a better way...
second_T_alm_unl = hp.synalm(ucl, lmax=lmax + dlmax, new=True)
second_lens_T_no_noise = lenspyx.alm2lenmap(second_T_alm_unl, [deflec_alm, None], nside, facres=facres)
hp.write_map("/home/amb257/rds/hpc-work/kappa_bispec/simulations/T_prime_phi_"+str(iterator)+".fits", second_lens_T_no_noise, overwrite=True)

#Now make sim with new unlensed T and new phi
#Make alm's for phi
second_phi_alm = hp.synalm(dimless_phi_cl, lmax=lmax + dlmax, new=True)
#Transform phi alm's into a deflection field. Multiply alm's by sqrt[l(l+1)]
second_deflec_alm = hp.almxfl(second_phi_alm, np.sqrt(np.arange(lmax + 1, dtype=float) * np.arange(1, lmax + 2)))
T_prime_phi_prime = lenspyx.alm2lenmap(second_T_alm_unl, [second_deflec_alm, None], nside, facres=facres)
hp.write_map("/home/amb257/rds/hpc-work/kappa_bispec/simulations/T_prime_phi_prime_"+str(iterator)+".fits", T_prime_phi_prime, overwrite=True)
