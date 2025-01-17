##Program to calculate the snr for the temperature based optimal estimator. Note must pass the Cl + Nl as power spec (N is the reconstruction noise for the power spectrum).
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('/home/amb257/software/cmplx_cmblensplus/wrap')
sys.path.append('/home/amb257/software/cmplx_cmblensplus/utils')
import basic
import curvedsky as cs
import camb

################ Parameters ###############

lmin = 5
lmax = 500
Tcmb  = 2.726e6    # CMB temperature
rlmin, rlmax = lmin, 3000 # CMB multipole range for reconstruction (ie l range of lensed cmb temp field use to reconstruct the lensing field)
nside = 2048
nbins = 20 # number bins for bispec estimator
bstype = 'equi'
cpmodel = 'modelw'
fitform = 'GM'
zcmb = 1088.69
zm   = 1.
zs   = [zcmb,zcmb,zcmb]
zmin, zmax = 0.0001, 40. #lensing signal dominated by intermediate redshifts.
zn = 50
btype = 'kkk'
z, dz = basic.bispec.zpoints(zmin,zmax,zn)

################# Matter Power spec ###########

### Pars (same as sims)
pars = camb.CAMBparams(min_l=1)
pars.set_cosmology(H0=67.32, ombh2=0.02238, omch2=0.12010)
pars.InitPower.set_params(As=2.1005e-9, ns=0.9660, r=0)
pars.set_for_lmax(5500, lens_potential_accuracy=1)
pars.set_matter_power(redshifts=[0.], kmax=30)
results = camb.get_results(pars)
k, __, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=30, npoints=1000)
s8 = np.array(results.get_sigma8())

################# Main code ####################

#for index, item in enumerate(snr_L):
 ################ Power spectra ################
ls, cl_unl, cl_len, phi_camb = np.loadtxt('/home/amb257/kappa_bispec/make_sims_parallel/camb_lencl_phi.txt')
L = np.arange(lmax+1)
L_fornorm = np.arange(rlmax+1)
Lfac = (L*(L+1.))**2/(2*np.pi)
lcl = cl_len[0:lmax+1] / Tcmb**2
lcl_fornorm = cl_len[0:rlmax+1] / Tcmb**2

#Make noise power spectra
theta_fwhm = 1.4 #In arcminutes
sigma_noise = 10 #in muK-arcmin
arcmin2radfactor = np.pi / 60.0 / 180.0
noise_cl = (sigma_noise*arcmin2radfactor/Tcmb)**2*np.exp(L*(L+1.)*(theta_fwhm*arcmin2radfactor)**2/np.log(2.)/8.)
noise_cl_fornorm = (sigma_noise*arcmin2radfactor/Tcmb)**2*np.exp(L_fornorm*(L_fornorm+1.)*(theta_fwhm*arcmin2radfactor)**2/np.log(2.)/8.)
ocl = np.copy(lcl) + noise_cl
ocl_fornorm = np.copy(lcl_fornorm) + noise_cl_fornorm

#Calculate the quadratic estimator normalisation. This is the N0 bias term to the power spectrum. We want to use the lensing potential power spectrum + N0 bias as the Cl in this variance calculation.
#Notice this is where the max l values of the lensed CMB come in that we use for reconstruction of the lensing field.
norm_phi, curl_norm = cs.norm_quad.qtt('lens',lmax,rlmin,rlmax,lcl_fornorm,ocl_fornorm, lfac='k')

#Create the power spec we want to use in the variance calculation - change lcl to ocl if want to include instrument noise. Since snr so low in temp only case we're doing everything without noise.
cl_var = ocl + norm_phi
snr = basic.bispec.bispeclens_snr(cpmodel,fitform,z,dz,zs,lmin,lmax,cl_var,k,pk[0],btype=btype, ro=5)

#np.savetxt('noise_snr_fold_rlmax3000.txt', (snr_L,snr))
print(snr)