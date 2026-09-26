"""Rerun the thesis bins with the corrected, physical signed N2 integral.

Preserves the original half-open integer bins, all labelled triangles,
Wigner-000 weights and bin normalisation. The closed-form factorial Wigner formula
removes a machine-specific library dependency. Vectorised triangle slices
reuse the same eight radial integrals instead of recomputing them per triangle.
"""
from __future__ import annotations
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import sys
import time
import numpy as np
from scipy.special import gammaln

CALCULATION = Path(__file__).resolve().parents[1] / 'full_n2_bias_calculation'
sys.path.insert(0, str(CALCULATION))
from full_N2 import do_N2_integral

BIN_EDGES = np.array([20, 40, 60, 80, 100, 200, 300, 400, 500,
                      600, 700, 800, 900, 1000])


def N(L1, L2, L3):
    """Original integer-multipole Wigner-000 weight using log factorials."""
    a, b, c = np.broadcast_arrays(np.asarray(L1, dtype=int),
                                  np.asarray(L2, dtype=int),
                                  np.asarray(L3, dtype=int))
    result = np.zeros(a.shape, dtype=float)
    allowed = ((a >= 0) & (b >= 0) & (c >= 0)
               & (a + b >= c) & (a + c >= b) & (b + c >= a)
               & ((a + b + c) % 2 == 0))
    av, bv, cv = a[allowed], b[allowed], c[allowed]
    g = (av + bv + cv) // 2
    log_square = (2 * (gammaln(g + 1) - gammaln(g - av + 1)
                      - gammaln(g - bv + 1) - gammaln(g - cv + 1))
                  + gammaln(2 * (g - av) + 1)
                  + gammaln(2 * (g - bv) + 1)
                  + gammaln(2 * (g - cv) + 1) - gammaln(2 * g + 2))
    result[allowed] = ((2. * av + 1) * (2. * bv + 1) * (2. * cv + 1)
                       * np.exp(log_square) / (4 * np.pi))
    return float(result) if result.ndim == 0 else result


def find_angles(L1, L2, L3):
    """Original triangle orientation, with round-off clipping."""
    L1, L2, L3 = np.broadcast_arrays(np.asarray(L1, dtype=float),
                                     np.asarray(L2, dtype=float),
                                     np.asarray(L3, dtype=float))
    theta1 = np.arccos(np.clip((L2**2 + L3**2 - L1**2) / (2 * L2 * L3), -1, 1))
    theta3 = np.arccos(np.clip((L1**2 + L2**2 - L3**2) / (2 * L1 * L2), -1, 1))
    return np.zeros_like(L1), np.pi - theta3, 2 * np.pi - (theta1 + theta3)


def _triangle_slice(lower, upper, L1, fold):
    divisor = 2 if fold else 1
    sides = np.arange(int(lower / divisor), int(upper / divisor))
    L2, L3 = np.meshgrid(sides, sides, indexing='ij')
    weights = N(L1, L2, L3)
    allowed = weights != 0
    return (np.full(np.count_nonzero(allowed), L1), L2[allowed], L3[allowed]), weights[allowed]


def N_bin(bin_edges, is_it_folded):
    return np.array([sum(np.sum(_triangle_slice(lo, hi, L1, is_it_folded)[1])
                         for L1 in range(int(lo), int(hi)))
                     for lo, hi in zip(bin_edges[:-1], bin_edges[1:])])


def _weighted_slice(task, config):
    lower, upper, L1, fold = task
    lengths, weights = _triangle_slice(lower, upper, L1, fold)
    if not len(weights):
        return 0., 0., 0
    angles = find_angles(*lengths)
    values = do_N2_integral(*lengths, *angles, config.cl_phi_interp,
                            config.ctot_interp, config.lcl_interp,
                            config.ctotprime_interp, config.lclprime_interp,
                            config.lcldoubleprime_interp, config.norm_factor_phi)
    return float(np.dot(weights, values)), float(np.sum(weights)), len(weights)


def process_L1(args):
    index, L1, edges, config = args
    return index, _weighted_slice((edges[index], edges[index + 1], L1, False), config)[0]


def process_L1_folded(args):
    index, L1, edges, config = args
    return index, _weighted_slice((edges[index], edges[index + 1], L1, True), config)[0]


def _worker_init(config):
    global _worker_config
    _worker_config = config


def _worker_slice(task):
    return _weighted_slice(task, _worker_config)


def bin_N2(bin_edges, config, fold=False, num_processes=1, diagnostics=None):
    """Physical signed bias averaged with the unchanged thesis weights."""
    answer = []
    pool = Pool(num_processes, initializer=_worker_init, initargs=(config,)) if num_processes and num_processes > 1 else None
    try:
        for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
            start = time.monotonic()
            tasks = [(lower, upper, L1, fold) for L1 in range(int(lower), int(upper))]
            rows = list(pool.map(_worker_slice, tasks)) if pool else [_weighted_slice(task, config) for task in tasks]
            numerator = sum(row[0] for row in rows)
            denominator = sum(row[1] for row in rows)
            if denominator <= 0:
                raise ValueError(f'No weighted triangles in bin [{lower}, {upper})')
            value = numerator / denominator
            answer.append(value)
            record = {'lower': int(lower), 'upper_exclusive': int(upper),
                      'nonzero_weight_labelled_triangles': int(sum(row[2] for row in rows)),
                      'normalisation': denominator, 'signed_kappa_bias': value}
            if diagnostics is not None:
                diagnostics.append(record)
            print(f'Bin [{lower}, {upper}): {value:.12e} ({time.monotonic() - start:.2f} s)', flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    return np.asarray(answer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folded', nargs='?', help='Legacy True/False option')
    parser.add_argument('--shape', choices=['both', 'equilateral', 'folded'], default='both')
    parser.add_argument('--processes', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'outputs')
    args = parser.parse_args()
    if args.folded is not None:
        args.shape = 'folded' if args.folded.lower() in ['true', 't', '1', 'yes', 'y'] else 'equilateral'
    from runtime_config import load_config
    config = load_config()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shapes = ['equilateral', 'folded'] if args.shape == 'both' else [args.shape]
    for shape in shapes:
        diagnostics = []
        values = bin_N2(BIN_EDGES, config, fold=shape == 'folded',
                        num_processes=args.processes, diagnostics=diagnostics)
        np.save(args.output_dir / f'Simple_N2_binned_{shape}.npy',
                ((BIN_EDGES[1:] + BIN_EDGES[:-1]) / 2, values))
        (args.output_dir / f'Simple_N2_binned_{shape}_checks.json').write_text(
            json.dumps({'shape': shape, 'sign': 'physical signed kappa bias',
                        'binning_rule': 'unchanged half-open integer bins and all labelled Wigner-weighted triangles',
                        'bins': diagnostics}, indent=2) + '\n')


if __name__ == '__main__':
    main()
