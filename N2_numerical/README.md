# Rerun the thesis N2 approximation

The current general evaluator uses repository-local spectra and a bundled,
verified TT reconstruction normalization. The numerical rerun needs Python,
NumPy and SciPy. It was checked with Python 3.12.3, NumPy 2.4.4 and SciPy 1.17.1.

Run these commands from the repository root:

```bash
python -m pip install numpy scipy
python N2_numerical/Binning_effects/full_n2_bias_calculation/full_N2.py
python N2_numerical/Binning_effects/binning/Simple_N2_binning.py --shape both --processes 1
```

The first calculation writes `equilN2_from_full_int.txt` and
`foldN2_from_full_int.txt` under `Binning_effects/outputs/`. It preserves existing
L grids; for a new output directory the defaults are L=2,...,999 for equilateral
and L=2,7,...,997 for folded configurations. The second calculation writes
`Simple_N2_binned_equilateral.npy` and `Simple_N2_binned_folded.npy` there.
Both commands accept `--output-dir PATH`; the binning command also accepts
`--shape equilateral` or `--shape folded`. JSON files record settings and checks.

The text and NumPy outputs have two rows: lensing multipole/bin centre, then
the **signed physical convergence bispectrum**. The unbinned equilateral bias
is positive and the folded bias is negative. Equilateral plotting uses the
saved values directly; folded magnitude plots use their absolute value.
Binned equilateral values can have either sign.

The evaluator includes each of the following once:

- The corrected physical sign and the three original routed contributions.
- The phi-to-kappa conversion, `prod(Li*(Li+1))/8`.
- The stated geometric prescription `prod((Li+1)/Li)`, with triangle angles
  unchanged. This is an additional prescription, not an exact full-sky result.

Eight radial spectral integrals are cached and combined for each triangle.
Composite Gauss–Legendre integration is split at interpolation knots; the final
combined result is checked between orders 8 and 12, with further refinement
available. Binning retains the original half-open integer bins, labelled
triangles, Wigner weights and bin normalization.

## Bundled normalization

[`phi_norm_lmax3000_noise10.npy`](Binning_effects/full_n2_bias_calculation/data/phi_norm_lmax3000_noise10.npy)
contains the independently computed full-sky TT normalization for external
**L=1 through 1001**; element zero is an unused zero placeholder. Evaluation
beyond that range raises an error instead of extrapolating.

The cache is specific to the supplied spectrum file, CMB reconstruction
multipoles 2–3000, 10 microkelvin-arcmin temperature noise, a 1.4 arcmin beam,
and the lensed TT response. Spectrum and cache hashes are checked at runtime.
See [provenance.json](Binning_effects/full_n2_bias_calculation/data/provenance.json)
for the formula, settings and verification. A different physical configuration
requires a newly computed normalization, not reuse of this cache.

The unbinned command accepts `--normalization-cache PATH` as an explicit
override for an identical verified cache. If the cache is absent, an installed
`curvedsky` can compute the normalization. Normal runs use only the files in
this repository; they have no dependency on the separate audit workspace.
