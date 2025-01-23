""" Code to  create a configuration file for all analyses in the pipeline
Defines all parameters and interpolates all CMB spectra used in bispectrum estimators and numerical approximations """

import numpy as np
from scipy.interpolate import interp1d
import sys, os
sys.path.append('/home/amb257/software/cmplx_cmblensplus/wrap')
sys.path.append('/home/amb257/software/cmplx_cmblensplus/utils')
import curvedsky as cs # for quadratic estimator normalisation

class CMBConfig:
    def __init__(self):
        # Default parameters
        self.Tcmb = 2.726e6  # CMB temperature in microkelvin
        self.rlmin = 2
        self.rlmax = 3000    # CMB multipole range for reconstruction
        self.nside = 2048
        self.nsims = 448     # Number of simulations to average over
        self.ellmin = 2
        self.ellmax = 3000
        
        # Instrument parameters
        self.theta_fwhm = 1.4  # arcminutes
        self.sigma_noise = 10  # muK-arcmin
            
        # Initialize derived quantities
        self._initialize_power_spectra()
    
    def _initialize_power_spectra(self):
        """Initialize all power spectra and interpolation functions"""
        # Load CAMB spectra
        self.ls, self.cl_unl, self.cl_len, self.cl_phi = np.loadtxt(
            '/home/amb257/kappa_bispec/make_sims_parallel/camb_lencl_phi.txt'
        )
        
        # Initialize L array and Lfac
        self.L = np.arange(self.rlmax + 1)
        self.Lfac = (self.L * (self.L + 1.) / 2)**2
        
        # Calculate various cls
        self.lcl = self.cl_len[0:self.rlmax + 1] / self.Tcmb**2
        self.ucl = self.cl_unl[0:self.rlmax + 1] / self.Tcmb**2
        self.cl_kappa = self.Lfac * self.cl_phi[0:self.rlmax + 1]
        self.cl_phi = self.cl_phi[0:self.rlmax + 1]
        
        # Calculate noise power spectra
        self.arcmin2radfactor = np.pi / 60.0 / 180.0
        self.noise_cl = (self.sigma_noise * self.arcmin2radfactor / self.Tcmb)**2 * \
                       np.exp(self.L * (self.L + 1.) * \
                       (self.theta_fwhm * self.arcmin2radfactor)**2 / np.log(2.) / 8.)
        
        self.ocl = np.copy(self.lcl) + self.noise_cl

        # Calculate normalisation factors
        self.phi_norm, self.phi_curl_norm = cs.norm_quad.qtt('lens', self.rlmax, self.rlmin, self.rlmax, self.lcl, self.ocl, lfac='')
        self.norm_factor_phi = interp1d(self.L, self.phi_norm, kind='cubic', bounds_error=False, fill_value="extrapolate")
        
        # Create interpolation functions
        self._initialize_interpolations()
    
    def _initialize_interpolations(self):
        """Initialize all interpolation functions"""
        interp_kwargs = {'kind': 'cubic', 'bounds_error': False, 'fill_value': "extrapolate"}
        
        # Basic interpolations
        self.cl_phi_interp = interp1d(self.L, self.cl_phi, **interp_kwargs)
        self.lcl_interp = interp1d(self.L, self.lcl, **interp_kwargs)
        self.ctot_interp = interp1d(self.L, self.ocl, **interp_kwargs)
        
        # Derivative interpolations
        self.ctotprime = np.gradient(self.ocl, self.L)
        self.ctotprime_interp = interp1d(self.L, self.ctotprime, **interp_kwargs)
        
        self.lclprime = np.gradient(self.lcl, self.L)
        self.lclprime_interp = interp1d(self.L, self.lclprime, **interp_kwargs)
        
        self.lcldoubleprime = np.gradient(self.lclprime, self.L)
        self.lcldoubleprime_interp = interp1d(self.L, self.lcldoubleprime, **interp_kwargs)


