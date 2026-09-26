import numpy as np
import camb

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

cl_unl=powers['unlensed_scalar'][:,0] #selects TT column
cl_len=powers['lensed_scalar'][:,0]
phi_cl=powers['lens_potential'][:,0]
ls = np.arange(cl_unl.shape[0])

np.savetxt('camb_lencl_phi.txt', (ls, cl_unl, cl_len, phi_cl))

