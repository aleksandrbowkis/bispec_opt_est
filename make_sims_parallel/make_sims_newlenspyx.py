#Make sets of simulations I, i, i' where
#I is an unlensed CMB T1, lensed by phi1 then noise added
#i is the same unlensed CMB T1, lensed by phi1 WITHOUT noise
#i' is a new unlensed CMB T2, lensed by the same phi1
#Then we make another set of three sims with different unlensed T and different phi.
import lenspyx
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
#All the helpy stuff we	need is	baked into lenspyx now so just use that
from lenspyx.utils_hp import synalm, almxfl, alm2cl
import healpy as hp #for writemap

################# parameters which impact of the accuracy of the result (and the execution time) ####################

nside = 2048 # The lensed tlm's are computed with healpy map2alm from a lensed map at resolution 'nside_lens'
Tcmb  = 2.726e6    # CMB temperature
lmax_len = 3000 # desired lmax of the lensed field.
dlmax = 1024  # lmax of the unlensed fields is lmax + dlmax.  (some buffer is required for accurate lensing at lmax)
epsilon = 1e-17 # target accuracy of the output maps (execution time has a fairly weak dependence on this)
lmax_unl, mmax_unl = lmax_len + dlmax, lmax_len + dlmax

###################### Read in Cl's ###############################################################################

_, cl_unl, _, phi_cl = np.loadtxt('camb_lencl_phi.txt')
ucl = cl_unl / Tcmb**2
ell = np.arange(lmax_unl)

#Make noise power spectra
theta_fwhm = 1.4 #In arcminutes
sigma_noise = 10 #in muK-arcmin
arcmin2radfactor = np.pi / 60.0 / 180.0
noise_cl = (sigma_noise*arcmin2radfactor/Tcmb)**2*np.exp(ell*(ell+1.)*(theta_fwhm*arcmin2radfactor)**2/np.log(2.)/8.)

################## Make simulations (gauss realisation of phi, unlensed T, lensed T)  ################

#Read in command line arg (n.b. the 0 element of the array is the program name).
iterator = int(sys.argv[1])

#All the helpy stuff we need is baked into lenspyx now so just use that
from lenspyx.utils_hp import synalm, almxfl, alm2cl

#Make alm's for an unlensed T using ucl
T_alm_unl = synalm(ucl, lmax=lmax_unl, mmax=mmax_unl)

#Make alm's for phi
phi_alm = synalm(phi_cl, lmax=lmax_unl, mmax=mmax_unl)

# We then transform the lensing potential into spin-1 deflection field, and deflect the temperature map.
dlm = almxfl(phi_alm, np.sqrt(np.arange(lmax_unl + 1, dtype=float) * np.arange(1, lmax_unl + 2)), None, False)

#Make noise map
noise_map = hp.synfast(noise_cl, nside, lmax=lmax_unl, mmax=mmax_unl)

# Geometry on which to produce the lensed map
geom_info = ('healpix', {'nside':2048}) # here we will use an Healpix grid with nside 2048

# Unlensed T map is this:
geom = lenspyx.get_geom(geom_info)
Tunl = geom.alm2map(T_alm_unl, lmax_unl, mmax_unl, nthreads=os.cpu_count())

# Now find lensed temp map
Tlen = lenspyx.alm2lenmap(T_alm_unl, dlm, geometry=geom_info, verbose=1)

#Add the noise map to the lensed T map
Tlen_noise = Tlen + noise_map

#Now make a simulation with a different unlensed T and lensed by the same phi
#Note just doing alm2lenmap again just spits out the same thing rather than a new one. Have to make a new map from T_alm_unl
#geom = lenspyx.get_geom(geom_info)
#Tunl_2 = geom.alm2map(T_alm_unl, lmax_unl, mmax_unl, nthreads=os.cpu_count())
#Tlen_2 = lenspyx.alm2lenmap(Tunl_2, dlm, geometry=geom_info, verbose=1)
#Tlen_2_noise = Tlen_2 + noise_map

#Now make sim with new unlensed T AND new phi
#phi_alm_2 = synalm(phi_cl, lmax=lmax_unl, mmax=mmax_unl)
#QQQQQQQQQQQQ Make phi_2 map - question is this the map that actually lenses the unl T tho... as made seperately and use the alms not the map so maybe save dlm?
#phi_map_2 = hp.synfast(phi_cl, nside, lmax=lmax_unl, mmax=mmax_unl)
#Transform phi alm's into a deflection field. Multiply alm's by sqrt[l(l+1)]
#dlm_2 = almxfl(phi_alm_2, np.sqrt(np.arange(lmax_unl + 1, dtype=float) * np.arange(1, lmax_unl + 2)), None, False)
#T_prime_phi_prime = lenspyx.alm2lenmap(Tunl_2, dlm_2, geometry=geom_info, verbose=1)
#T_prime_phi_prime += noise_map

# Save the maps - we save phi and lensed noisy temperature map
#currently testing that everything works for the power spec
np.savetxt("/home/amb257/kappa_bispec/optimal_est/powerspec_scatter/testsim/phi"+str(iterator)+".txt", phi_alm) #Phi alms
#/home/amb257/rds/hpc-work/kappa_bispec/simulations/phi"+str(iterator)+".txt", phi_alm) #Phi alms
hp.write_map("/home/amb257/kappa_bispec/optimal_est/powerspec_scatter/testsim/Tlen_noise"+str(iterator)+".fits", Tlen_noise, overwrite=True) #lensed temp noise


#/home/amb257/rds/hpc-work/kappa_bispec/simulations/T_phi_noise_"+str(iterator)+".fits", Tlen_noise, overwrite=True) #Lensed Temp noise
#hp.write_map("/home/amb257/rds/hpc-work/kappa_bispec/simulations/T_prime_phi_noise"+str(iterator)+".fits", Tlen_2_noise, overwrite=True) #Lensed Temp noise
#hp.write_map("/home/amb257/rds/hpc-work/kappa_bispec/simulations/tests/phi_2.fits", phi_map_2, overwrite=True) #Phi_2 map
#hp.write_map("/home/amb257/rds/hpc-work/kappa_bispec/simulations/T_prime_phi_prime_"+str(iterator)+".fits", T_prime_phi_prime, overwrite=True) #Lensed Temp noise
