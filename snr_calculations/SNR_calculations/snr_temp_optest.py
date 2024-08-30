##Program to calculate the snr for the temperature based optimal estimator. Note must pass the Cl + Nl as power spec (N is the reconstruction noise for the power spectrum).
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('/home/amb257/software/lensplus_nolss/cmblensplus/wrap')
sys.path.append('/home/amb257/software/lensplus_nolss/cmblensplus/utils')
import basic
import curvedsky as cs
import camb

################ Parameters ###############

lmin = 2
lmax = 2000
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

################ Power spectra ################

################ FOR MV ONLY ##################

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams(min_l=1)
pars.set_cosmology(H0=67.32, ombh2=0.02238, omch2=0.12010)
pars.InitPower.set_params(As=2.1005e-9, ns=0.9660, r=0)
pars.set_for_lmax(5500, lens_potential_accuracy=1);
print(pars)

#calculate results for these parameters
results = camb.get_results(pars)
#get dictionary of CAMB power spectra
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True) #output cl not dl

cl_unl=powers['unlensed_scalar']
cl_len=powers['lensed_scalar']
ls = np.arange(cl_unl.shape[0])
L = np.arange(lmax+1)
L_fornorm = np.arange(rlmax+1)
Lfac = (L*(L+1.))**2/(2*np.pi)
lcl_fornorm = cl_len[0:rlmax+1] / Tcmb**2
cl_phi=powers['lens_potential'][:,0]
cl_kappa = cl_phi[0:lmax+1] / Lfac[0:lmax+1]

#Make noise power spectra for S3 wide
theta_fwhm = 1 #1.4 #In arcminutes
sigma_noise = 6 #10 #in muK-arcmin
arcmin2radfactor = np.pi / 60.0 / 180.0
noise_cl_fornorm = (sigma_noise*arcmin2radfactor/Tcmb)**2*np.exp(L_fornorm*(L_fornorm+1.)*(theta_fwhm*arcmin2radfactor)**2/np.log(2.)/8.)
ocl_fornorm = lcl_fornorm + noise_cl_fornorm

############### FOR TT ONLY ##################

ls, cl_unl, cl_len, cl_phi = np.loadtxt('/home/amb257/kappa_bispec/make_sims_parallel/camb_lencl_phi.txt')
L = np.arange(lmax+1)
L_fornorm = np.arange(rlmax+1)
Lfac = (L*(L+1.))**2/(2*np.pi)
#print(np.shape(Lfac))
cl_kappa = cl_phi[0:lmax+1] / Lfac[0:lmax+1]
lcl_fornorm = cl_len[0:rlmax+1] / Tcmb**2

#Make noise power spectra for S3 wide
theta_fwhm = 1 #1.4 #In arcminutes
sigma_noise = 6 #10 #in muK-arcmin
arcmin2radfactor = np.pi / 60.0 / 180.0
noise_cl_fornorm = (sigma_noise*arcmin2radfactor/Tcmb)**2*np.exp(L_fornorm*(L_fornorm+1.)*(theta_fwhm*arcmin2radfactor)**2/np.log(2.)/8.)
ocl_fornorm = lcl_fornorm + noise_cl_fornorm

################# QE NORM ###################

#Calculate the quadratic estimator normalisation. This is the N0 bias term to the power spectrum. We want to use the lensing potential power spectrum + N0 bias as the Cl in this variance calculation.
#Notice this is where the max l values of the lensed CMB come in that we use for reconstruction of the lensing field.
#NOTE - change second ps to ocl_fornorm if want noise - this is only place it enters.

QDO = [True,True,True,True,True,False]
#Compute norm for all qest. Ag is norm for gradient, Ac for curl. Wg and Wc are the weights in the MV QE (linear combo of the TT, TE etc estimators)
Ag, Ac, Wg, Wc = cs.norm_quad.qall('lens',QDO,lmax,rlmin,rlmax,lcl_fornorm,ocl_fornorm)

#norm_phi_MV = Ag[5,:,None]

#norm_phi, curl_norm = cs.norm_quad.qtt('lens',lmax,rlmin,rlmax,lcl_fornorm,ocl_fornorm, lfac='k')

################# Main code ####################

cl_var = cl_kappa + norm_phi_MV[0:lmax+1]
snr = basic.bispec.bispeclens_snr(cpmodel,fitform,z,dz,zs,lmin,lmax,cl_var,k,pk[0],btype=btype)
