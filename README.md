# Optimal Estimator for the Reconstructed CMB Lensing Bispectrum  

This repository contains code for computing the optimal estimator of the reconstructed Cosmic Microwave Background (CMB) lensing bispectrum. This work is part of my PhD research and is under active development.

The updated N2 numerical calculation can be rerun using repository-local data
with NumPy and SciPy. See [N2 rerun instructions](N2_numerical/README.md) for
commands, output sign conventions and the bundled normalization's limits.

## **Key Requirements**  
To eventually run this code, the following libraries and dependencies must be installed:  

- **[cmblensplus](https://github.com/toshiyan/cmblensplus)**: A toolkit for CMB lensing analysis.  
- **[CAMB](https://camb.info/)**: Code for calculating CMB power spectra.  
- **[lenspyx](https://github.com/carronj/lenspyx)**: Lensed CMB maps simulation package

## **Current Status**  
- **Development**: The code is currently under construction. 
- **Usability**: It is not yet user-friendly. Please wait for future updates or contact me for specific questions related to its use.  

## **Pipeline overview (thesis Chapter 4, Section 4.6)**

1. **Simulation generation** — `make_sims_parallel/make_sims_newlenspyx.py` (+ `makecls_camb.py` for the input CAMB spectra `camb_lencl_phi.txt`). Produces 500 full-sky Gaussian-lensed CMB temperature maps (Gaussian unlensed T lensed by a Gaussian φ via `lenspyx`, with Simons Observatory LAT-like noise: 10 μK-arcmin, 1.4′ beam). Output `.fits` maps are large (~1 TB total) and are **not** in this repo — they live on HPC scratch storage and must be regenerated.
2. **Estimator evaluation** — `cmplx_estimator/k6_cmplx.py`. For a "data" map and a cyclic grouping of 3 further sims, builds the quadratic-estimator reconstructions, forms the factorised complex realisation-dependent estimator, and bins the bispectrum (`cmblensplus`). Set `bstype = 'fold'` or `'equi'` at the top of the script to switch configuration; submitted via `cmplx_estimator/cmplx.sub` as a SLURM job array over all (data, sim-start) pairs.
3. **Averaging over groupings** — `cmplx_estimator/average.py` (equilateral) / `average_fold.py` (folded). Averages the per-grouping outputs from step 2 for a given data realisation, and computes the std-dev across groupings, producing `cmplx_bispec_data{i}.txt` / `fold_cmplx_bispec_data{i}.txt`.
4. **Averaging over data realisations + error bars** — `Plots/paper_plots/equi_opt_est.ipynb`, cell 1. Loads all 100 per-realisation files from `Plots/data/data_for_paper/{equi_opt_est,fold_opt_est}/`, takes the mean (the plotted estimator curve) and `std/sqrt(100)` (the plotted error bars — the standard error of the mean across the 100 data realisations).
5. **Numerical N₍RD₎⁽²⁾ bias** — `N2_numerical/Binning_effects/full_n2_bias_calculation/full_N2.py` numerically integrates the general low-$L$ approximation for arbitrary triangle shape, specialised to equilateral/folded in `__main__`, producing `equilN2_from_full_int.txt`/`foldN2_from_full_int.txt` (the analytic comparison curve in Figs. 4.5–4.7). `N2_numerical/Binning_effects/binning/Simple_N2_binning.py` produces the binned comparison arrays used in Figs. 4.10–4.11. See the [N2 instructions](N2_numerical/README.md) for the current runnable pipeline.
6. **Theoretical covariance** — `Errors/Variance_theory.py` evaluates the Gaussian binned-bispectrum covariance formula directly, producing `Errors/newequi_var.txt`/`newfold_var.txt` (theory curve in Fig. 4.7/4.8).

All the intermediate/final **result data** needed to reproduce the published plots (the 100+100 per-realisation bispectrum files, N0/N1/N2 numerical results, theoretical variance, correlation matrices) are committed under `Plots/data/`, `N0_numerical/`, `N1_numerical/`, `N2_numerical/`, and `Errors/`. Only the raw simulated maps and the very large intermediate per-grouping estimator outputs (step 2, before averaging) are excluded.

### Known gaps / things to check after cloning to a new machine
- Most scripts use **hard-coded absolute paths** under `/home/amb257/...` (e.g. `/home/amb257/software/cmplx_cmblensplus`, `/home/amb257/kappa_bispec/make_sims_parallel/camb_lencl_phi.txt`, `/home/amb257/rds/hpc-work/...`). Either recreate this directory layout on the new machine or update the paths.
- The modified `cmblensplus` fork at `/home/amb257/software/cmplx_cmblensplus` (used for complex-field bispectrum estimation) is a separate dependency and is not part of this repo.
- `Plots/paper_plots/equi_opt_est.ipynb` cells 9–13 are exploratory/debugging cells (not used for any published figure) that reference a few additional HPC-scratch-only files (`onesimterms/results/direct_bsp_448_sims.txt`, `bias_from_sims/bispec/N0tests/unlensed_sim_N0_kappa.txt`, per-sim `initialterm_{i}.txt` files); these were not copied in.
- `Plots/paper_plots/Ch3_plots.ipynb` and `FoldedN0_paperstyle.ipynb` (Chapter 3 plots) reference some comparison data from sibling, non-git directories (`kappa_bispec/alba_2023`, `kappa_bispec/CMBlens_bispectrum_noisebiases`, `kappa_bispec/alba_code`) and HPC project storage (`rds-dirac-dp002`) that are external/collaborator outputs and are not included here.

## **Contact**  
If you have any questions, please feel free to reach out:  
- **Name**: Aleksandr Bowkis
- **Email**: amb257@cam.ac.uk 
- **Institution**: University of Cambridge Institute of Astronomy
