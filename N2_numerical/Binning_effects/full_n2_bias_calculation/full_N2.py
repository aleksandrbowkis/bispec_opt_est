"""Signed low-L N2 approximation to the reconstructed convergence bispectrum.

The original three routed permutations are retained. The additional geometric
prescription multiplies the flat-sky result by prod_i[(L_i+1)/L_i]; it does not
change the angles and is separate from the existing phi-to-kappa conversion.
Radial integration is factored into eight spectral moments, evaluated with
composite Gaussian quadrature and checked by increasing the quadrature order.
Importing this module performs no configuration, integration, or file writing.
"""

import numpy as np
from functools import lru_cache
from scipy.special import roots_legendre


def geometry_factor(L1, L2, L3):
    """Explicit shape-preserving L(L+1) prescription; not a kappa conversion."""
    L1, L2, L3 = np.broadcast_arrays(L1, L2, L3)
    if np.any(L1 <= 0) or np.any(L2 <= 0) or np.any(L3 <= 0):
        raise ValueError('External lensing multipoles must be positive.')
    return ((L1 + 1) / L1) * ((L2 + 1) / L2) * ((L3 + 1) / L3)

#### Integration functions ####

def calculate_n2_integrand(l, L1, L2, L3, x1, x2, x3, cphi_interp, ctot_interp, lcl_interp, 
                     ctotprime_interp, lclprime_interp, lcldoubleprime_interp, norm_phi_interp):
    """
    Calculate the N2 bias to the reconstructed lensing bispectrum for any configuration
    Note this returns only one permutation the remaining two can be found by: L1<->L2 and L1<->L3
    
    Args:
        l, L1, L2, L3: Multipole values
        x1, x2, x3: Angle of each triangle side
        ctot_interp: Interpolated total power spectrum (signal + noise)
        lcl_interp: Interpolated lensed CMB power spectrum
        ctotprime_interp: Interpolated derivative of total power spectrum
        lclprime_interp: Interpolated derivative of lensed power spectrum
        lcldoubleprime_interp: Interpolated second derivative of lensed power spectrum
    
    Returns:
        float or array: one routed contribution to the SIGNED physical kappa bias.
    """
    # Get interpolated values
    Ctot = ctot_interp(l)
    Ctt = lcl_interp(l)
    Ctot_prime = ctotprime_interp(l)
    Ctt_prime = lclprime_interp(l)
    Ctt_doubleprime = lcldoubleprime_interp(l)
    
    # Precompute some common cosine terms
    cos_x1_m_x2 = np.cos(x1 - x2)
    cos_x1_p_x2_m_2x3 = np.cos(x1 + x2 - 2*x3)
    cos_x2_m_x3 = np.cos(x2 - x3)
    cos_3x1_m_x2_m_2x3 = np.cos(3*x1 - x2 - 2*x3)
    cos_2x1_p_x2_m_3x3 = np.cos(2*x1 + x2 - 3*x3)
    cos_2x1_m_x2_m_x3 = np.cos(2*x1 - x2 - x3)
    
    # First term 
    term1 = -2 * l * L1 * Ctot_prime * (
        8 * (3*cos_x1_m_x2 + cos_x1_p_x2_m_2x3) * Ctt**2 +
        2 * l * (13*cos_x1_m_x2 + 5*cos_x1_p_x2_m_2x3) * Ctt * Ctt_prime +
        l**2 * (6*cos_x1_m_x2 + cos_3x1_m_x2_m_2x3 + 
                3*cos_x1_p_x2_m_2x3) * Ctt_prime**2
    )
    
    # Second term 
    term2 = -Ctot * (
        64 * L3 * cos_x2_m_x3 * Ctt**2 -
        2 * l * Ctt * (
            (27*L1*cos_x1_m_x2 + 9*L1*cos_x1_p_x2_m_2x3 - 
             50*L3*cos_x2_m_x3) * Ctt_prime +
            3 * l * (3*L1*cos_x1_m_x2 + L1*cos_x1_p_x2_m_2x3 - 
                     2*L3*cos_x2_m_x3) * Ctt_doubleprime
        ) +
        l**2 * Ctt_prime * (
            (-18*L1*cos_x1_m_x2 + 3*L3*cos_2x1_p_x2_m_3x3 +
             L1*cos_3x1_m_x2_m_2x3 - 9*L1*cos_x1_p_x2_m_2x3 +
             13*L3*cos_2x1_m_x2_m_x3 + 34*L3*cos_x2_m_x3) * Ctt_prime +
            l * (-6*L1*cos_x1_m_x2 + L3*cos_2x1_p_x2_m_3x3 -
                 L1*cos_3x1_m_x2_m_2x3 - 3*L1*cos_x1_p_x2_m_2x3 +
                 3*L3*cos_2x1_m_x2_m_x3 + 6*L3*cos_x2_m_x3) * Ctt_doubleprime
        )
    )
    
    # Combine terms with prefactor
    # Physical Type-A + Type-B sign; the archived evaluator had the opposite sign.
    prefactor = 1 / (128 * np.pi * Ctot**3)
    kappa_factor = L1*(L1+1) * L2*(L2+1) * L3*(L3+1) / 8
    
    result = geometry_factor(L1, L2, L3) * kappa_factor * prefactor * l * L1**2 * L2 * L3**2 * norm_phi_interp(L1) * cphi_interp(L2) * cphi_interp(L3) * (term1 + term2)
    
    return result

def _placement_coefficients(L1, L2, L3, x1, x2, x3, cphi, norm):
    """Eight coefficients of one routed integrand, with physical sign."""
    a = np.cos(x1 - x2)
    b = np.cos(x1 + x2 - 2*x3)
    c = np.cos(3*x1 - x2 - 2*x3)
    d = np.cos(x2 - x3)
    e = np.cos(2*x1 + x2 - 3*x3)
    f = np.cos(2*x1 - x2 - x3)
    coefficients = np.stack([
        -64*L3*d,
        -16*L1*(3*a + b),
        2*(27*L1*a + 9*L1*b - 50*L3*d),
        -4*L1*(13*a + 5*b),
        6*(3*L1*a + L1*b - 2*L3*d),
        18*L1*a - 3*L3*e - L1*c + 9*L1*b - 13*L3*f - 34*L3*d,
        -2*L1*(6*a + c + 3*b),
        6*L1*a - L3*e + L1*c + 3*L1*b - 3*L3*f - 6*L3*d])
    return coefficients * (L1**2 * L2 * L3**2 * norm(L1) * cphi(L2) * cphi(L3))


def geometry_coefficients(L1, L2, L3, x1, x2, x3, cl_phi_interp, norm_factor_phi):
    """Return shape (8, ...) coefficients; triangle inputs may be arrays.

    Includes all three original routed placements, the physical sign, one
    phi-to-kappa conversion and one explicit geometric prescription.
    """
    L1, L2, L3, x1, x2, x3 = np.broadcast_arrays(L1, L2, L3, x1, x2, x3)
    coefficients = (
        _placement_coefficients(L1, L2, L3, x1, x2, x3, cl_phi_interp, norm_factor_phi)
        + _placement_coefficients(L2, L1, L3, x2, x1, x3, cl_phi_interp, norm_factor_phi)
        + _placement_coefficients(L3, L2, L1, x3, x2, x1, cl_phi_interp, norm_factor_phi))
    kappa_factor = L1*(L1+1) * L2*(L2+1) * L3*(L3+1) / 8
    return coefficients * kappa_factor * geometry_factor(L1, L2, L3) / (128*np.pi)


@lru_cache(maxsize=32)
def _radial_moments_cached(ctot_interp, lcl_interp, ctotprime_interp,
                           lclprime_interp, lcldoubleprime_interp,
                           ellmin, ellmax, order):
    if not 0 < ellmin < ellmax:
        raise ValueError('Require 0 < ellmin < ellmax.')
    functions = (ctot_interp, lcl_interp, ctotprime_interp,
                 lclprime_interp, lcldoubleprime_interp)
    knots = [np.array([ellmin, ellmax])]
    for function in functions:
        if hasattr(function, 'x'):
            x = np.asarray(function.x)
            knots.append(x[(x > ellmin) & (x < ellmax)])
    # Also split at integer multipoles, including for callable-only spectra.
    knots.append(np.arange(np.ceil(ellmin), ellmax))
    knots = np.unique(np.concatenate(knots))
    x, w = roots_legendre(order)
    half_width = np.diff(knots)/2
    l = ((knots[:-1] + half_width)[:, None] + half_width[:, None]*x).ravel()
    weights = (half_width[:, None]*w).ravel()
    D, C, Dp, Cp, Cpp = [function(l) for function in functions]
    # Basis: l/D^3 times the following spectral monomials, integrated in dl.
    terms = np.array([D*C*C, l*Dp*C*C, l*D*C*Cp, l*l*Dp*C*Cp,
                      l*l*D*C*Cpp, l*l*D*Cp*Cp, l**3*Dp*Cp*Cp,
                      l**3*D*Cp*Cpp])
    moments = np.sum(terms * (l*weights/D**3)[None, :], axis=1)
    if not np.all(np.isfinite(moments)):
        raise FloatingPointError('Non-finite N2 radial spectral moment.')
    moments.setflags(write=False)
    return moments


def radial_moments(ctot_interp, lcl_interp, ctotprime_interp,
                   lclprime_interp, lcldoubleprime_interp,
                   ellmin=2, ellmax=3000, order=8):
    """Cached eight radial spectral integrals, independent of triangle shape.

    Basis ordering is [DC², lD'C², lDCC', l²D'CC', l²DCC'',
    l²D(C')², l³D'(C')², l³DC'C''], with common l/D³ measure.
    Keep the interpolation callables immutable while using this cache.
    """
    return _radial_moments_cached(ctot_interp, lcl_interp, ctotprime_interp,
                                  lclprime_interp, lcldoubleprime_interp,
                                  float(ellmin), float(ellmax), int(order))


def clear_radial_cache():
    """Call after modifying the arrays inside existing interpolation objects."""
    _radial_moments_cached.cache_clear()


def do_N2_integral(L1, L2, L3, x1, x2, x3, cl_phi_interp, ctot_interp,
                   lcl_interp, ctotprime_interp, lclprime_interp,
                   lcldoubleprime_interp, norm_factor_phi,
                   ellmin=2, ellmax=3000, *, rtol=1e-8,
                   quadrature_orders=(8, 12, 16, 24)):
    """Convergence-checked signed kappa bias for scalar or array triangles.

    The existing positional API is preserved. Eight spectral moments are
    cached, so millions of triangles can be evaluated without repeating radial
    quadrature. The final summed result is checked, including cancellations.
    A float-roundoff floor proportional to absolute contributions avoids a
    meaningless relative-accuracy demand at a zero of the approximation.
    """
    if rtol <= 0 or len(quadrature_orders) < 2:
        raise ValueError('Use positive rtol and at least two quadrature orders.')
    coefficients = geometry_coefficients(L1, L2, L3, x1, x2, x3,
                                         cl_phi_interp, norm_factor_phi)
    previous = None
    for order in quadrature_orders:
        moments = radial_moments(ctot_interp, lcl_interp, ctotprime_interp,
                                 lclprime_interp, lcldoubleprime_interp,
                                 ellmin, ellmax, order)
        result = np.einsum('i...,i->...', coefficients, moments)
        if previous is not None:
            absolute_scale = np.einsum('i...,i->...', np.abs(coefficients), np.abs(moments))
            tolerance = rtol*np.abs(result) + 256*np.finfo(float).eps*absolute_scale
            if np.all(np.abs(result - previous) <= tolerance):
                return result.item() if result.ndim == 0 else result
        previous = result
    raise RuntimeError('N2 radial integration did not converge at requested accuracy.')


def main(argv=None):
    """Regenerate the two thesis unbinned files, preserving their saved L grids."""
    import argparse
    import hashlib
    import json
    from pathlib import Path
    try:
        from .runtime_config import CMBConfig
    except ImportError:
        from runtime_config import CMBConfig
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--normalization-cache', type=Path)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parents[1]/'outputs')
    args = parser.parse_args(argv)
    config = CMBConfig(normalization_cache=args.normalization_cache)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {'output_convention': 'signed physical kappa bispectrum',
                'geometry': 'prod_i[(L_i+1)/L_i], applied once; kappa conversion already included',
                'quadrature': 'composite Gauss-Legendre on interpolation intervals, final sum checked at orders 8 and 12 (16/24 fallback)',
                'normalization': config.normalization_provenance, 'outputs': {}}
    for shape, name, fallback_grid in [
            ('equilateral', 'equilN2_from_full_int.txt', np.arange(2, 1000)),
            ('folded', 'foldN2_from_full_int.txt', np.arange(2, 1000, 5))]:
        path = args.output_dir/name
        L = np.loadtxt(path)[0] if path.exists() else fallback_grid
        if shape == 'equilateral':
            triangle = (L, L, L, 0, 2*np.pi/3, 4*np.pi/3)
        else:
            triangle = (L, L/2, L/2, 0, np.pi, np.pi)
        values = do_N2_integral(*triangle, config.cl_phi_interp, config.ctot_interp,
                                 config.lcl_interp, config.ctotprime_interp,
                                 config.lclprime_interp, config.lcldoubleprime_interp,
                                 config.norm_factor_phi, config.ellmin, config.ellmax)
        np.savetxt(path, np.vstack([L, values]))
        metadata['outputs'][name] = {'shape': shape, 'points': len(L),
            'first_L': float(L[0]), 'last_L': float(L[-1]),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    metadata_path = args.output_dir/'N2_recalculation_metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
