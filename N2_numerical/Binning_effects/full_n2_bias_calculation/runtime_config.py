"""Portable configuration for rerunning the thesis N2 calculation.

Physical settings and spectral interpolation match Configuration/config.py.
The bundled full-sky TT normalization cache avoids requiring curvedsky.
Its spectrum and cache hashes are checked before use. If no cache is available,
an installed curvedsky can compute the same normalization.
"""
from pathlib import Path
import hashlib
import numpy as np
from scipy.interpolate import interp1d

REPO = Path(__file__).resolve().parents[3]
VERIFIED_SPECTRUM_SHA256 = 'cc40c4abb4a5696db400ea34d62be706477a8b614c5fde2f746db79fb7855929'
VERIFIED_NORM_SHA256 = 'ec83da3dbaf3358b3777719526b3cfb87b373f76623afa62284f8649cffea0d6'


class CMBConfig:
    def __init__(self, normalization_cache=None):
        self.Tcmb = 2.726e6
        self.rlmin, self.rlmax = 2, 3000
        self.ellmin, self.ellmax = 2, 3000
        self.nside, self.nsims = 2048, 448
        self.theta_fwhm, self.sigma_noise = 1.4, 10
        spectrum_path = REPO / 'make_sims_parallel/camb_lencl_phi.txt'
        spectrum_hash = hashlib.sha256(spectrum_path.read_bytes()).hexdigest()
        self.ls, self.cl_unl, self.cl_len, phi = np.loadtxt(spectrum_path)
        self.L = np.arange(self.rlmax + 1)
        self.Lfac = (self.L * (self.L + 1.) / 2)**2
        self.lcl = self.cl_len[:self.rlmax + 1] / self.Tcmb**2
        self.ucl = self.cl_unl[:self.rlmax + 1] / self.Tcmb**2
        self.cl_phi = phi[:self.rlmax + 1]
        self.cl_kappa = self.Lfac * self.cl_phi
        self.arcmin2radfactor = np.pi / 60 / 180
        self.noise_cl = (self.sigma_noise * self.arcmin2radfactor / self.Tcmb)**2 * np.exp(
            self.L * (self.L + 1.) * (self.theta_fwhm * self.arcmin2radfactor)**2 / np.log(2.) / 8.)
        self.ocl = self.lcl.copy() + self.noise_cl
        kwargs = dict(kind='cubic', bounds_error=False, fill_value='extrapolate')
        self.cl_phi_interp = interp1d(self.L, self.cl_phi, **kwargs)
        self.lcl_interp = interp1d(self.L, self.lcl, **kwargs)
        self.ctot_interp = interp1d(self.L, self.ocl, **kwargs)
        self.ctotprime = np.gradient(self.ocl, self.L)
        self.lclprime = np.gradient(self.lcl, self.L)
        self.lcldoubleprime = np.gradient(self.lclprime, self.L)
        self.ctotprime_interp = interp1d(self.L, self.ctotprime, **kwargs)
        self.lclprime_interp = interp1d(self.L, self.lclprime, **kwargs)
        self.lcldoubleprime_interp = interp1d(self.L, self.lcldoubleprime, **kwargs)

        if normalization_cache is None:
            normalization_cache = Path(__file__).resolve().parent / 'data/phi_norm_lmax3000_noise10.npy'
        cache = Path(normalization_cache)
        if cache.exists():
            cache_hash = hashlib.sha256(cache.read_bytes()).hexdigest()
            if spectrum_hash != VERIFIED_SPECTRUM_SHA256 or cache_hash != VERIFIED_NORM_SHA256:
                raise ValueError('The supplied spectra/normalization cache do not match the verified thesis configuration.')
            self.phi_norm = np.load(cache)
            self.phi_curl_norm = None
            norm_L = np.arange(len(self.phi_norm))
            try:
                cache_label = str(cache.resolve().relative_to(REPO))
                cache_path_base = 'repository root'
            except ValueError:
                cache_label = str(cache.resolve())
                cache_path_base = 'absolute path (explicit override)'
            self.normalization_provenance = {
                'method': 'verified independent full-sky TT Wigner-3j sum',
                'cache': cache_label, 'cache_path_base': cache_path_base,
                'cache_sha256': cache_hash,
                'spectrum_sha256': spectrum_hash,
                'available_lensing_L': [0, int(norm_L[-1])],
                'reconstruction_ell_range': [self.rlmin, self.rlmax],
                'noise_microkelvin_arcmin': self.sigma_noise,
                'beam_arcmin': self.theta_fwhm}
        else:
            try:
                import curvedsky as cs
            except ImportError as exc:
                raise RuntimeError(
                    'The verified normalization cache is missing and curvedsky is unavailable. '
                    'Restore full_n2_bias_calculation/data/phi_norm_lmax3000_noise10.npy '
                    'or install curvedsky.') from exc
            self.phi_norm, self.phi_curl_norm = cs.norm_quad.qtt(
                'lens', self.rlmax, self.rlmin, self.rlmax, self.lcl, self.ocl, lfac='')
            norm_L = self.L
            self.normalization_provenance = {
                'method': 'curvedsky.norm_quad.qtt',
                'spectrum_sha256': spectrum_hash,
                'reconstruction_ell_range': [self.rlmin, self.rlmax]}
        # Do not extrapolate the short verified cache beyond its computed range.
        self.norm_factor_phi = interp1d(norm_L, self.phi_norm, kind='cubic', bounds_error=True)


def load_config(normalization_cache=None):
    """Return the original config attribute names with portable input paths."""
    return CMBConfig(normalization_cache=normalization_cache)
