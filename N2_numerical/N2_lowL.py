#### Import modules
import numpy as np
import camb
import vegas #ceuadmin/VEGAS2/2.01.17  is the module on HPC
from scipy.interpolate import interp1d

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

ls, cl_unl, cl_len, cl_phi = np.loadtxt('/home/amb257/kappa_bispec/make_sims_parallel/camb_lencl_phi.txt')
L = np.arange(rlmax+1)
Lfac = (L*(L+1.) / 2 )**2
lcl = cl_len[0:rlmax+1] / Tcmb**2
ucl = cl_unl[0:rlmax+1] / Tcmb**2 #dimless unlensed T Cl
cl_kappa = Lfac * cl_phi

#Make noise power spectra
theta_fwhm = 1.4 #In arcminutes
sigma_noise = 10 #in muK-arcmin
arcmin2radfactor = np.pi / 60.0 / 180.0
noise_cl = (sigma_noise*arcmin2radfactor/Tcmb)**2*np.exp(L*(L+1.)*(theta_fwhm*arcmin2radfactor)**2/np.log(2.)/8.)
ocl = np.copy(lcl) + noise_cl