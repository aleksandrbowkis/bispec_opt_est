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

cl_unl=powers['unlensed_scalar']
cl_len=powers['lensed_scalar']
ls = np.arange(cl_unl.shape[0])
ls_all = np.zeros((cl_unl.shape[0], 4))
for i in range(4):
    ls_all[:,i] = ls
print(np.shape(ls_all), np.shape(cl_unl))
np.savetxt('camb_MV_cls.txt', (ls_all, cl_unl))#, cl_unl[:,1], cl_unl[:,2], cl_unl[:,3], cl_len[:,0], cl_len[:,1], cl_len[:,2], cl_len[:,3]))


