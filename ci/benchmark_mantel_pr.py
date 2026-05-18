"""Benchmark Mantel Pearson Cython vs Numba paths.

This script compares the Cython and Numba Mantel helper paths from the same
checkout by toggling ``_mantel.NUMBA_AVAILABLE``.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time

import numpy as np
from scipy.spatial.distance import squareform

import skbio
from skbio import DistanceMatrix
from skbio.stats.distance import _mantel


def _numba_info() -> tuple[str, str]:
    try:
        import numba

        return numba.__version__, str(numba.get_num_threads())
    except Exception:
        return "not installed", "not available"


def print_metadata() -> None:
    numba_version, numba_threads = _numba_info()
    print("# Metadata")
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")
    print(f"machine={platform.machine()}")
    print(f"processor={platform.processor() or 'unknown'}")
    print(f"logical_cores={os.cpu_count()}")
    print(f"numpy={np.__version__}")
    print(f"skbio={skbio.__version__}")
    print(f"numba={numba_version}")
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}")
    print(f"NUMBA_NUM_THREADS={os.environ.get('NUMBA_NUM_THREADS', 'unset')}")
    print(f"numba_threads={numba_threads}")
    print(f"numba_available_in_module={_mantel.NUMBA_AVAILABLE}")
    print()


def make_full_case(n: int, seed: int) -> tuple[DistanceMatrix, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.random((n, n), dtype=np.float64)
    x = (x + x.T) * 0.5
    np.fill_diagonal(x, 0.0)

    y = rng.random((n, n), dtype=np.float64)
    y = (y + y.T) * 0.5
    np.fill_diagonal(y, 0.0)
    return DistanceMatrix(x), squareform(y, checks=False)


def make_condensed_case(n: int, seed: int) -> tuple[DistanceMatrix, np.ndarray]:
    rng = np.random.default_rng(seed)
    size = n * (n - 1) // 2
    x = rng.random(size, dtype=np.float64)
    y = rng.random(size, dtype=np.float64)
    return DistanceMatrix(x, condensed=True), y


def run_once(
    x: DistanceMatrix,
    y_flat: np.ndarray,
    permutations: int,
    seed: int,
    use_numba: bool,
):
    old_numba_available = _mantel.NUMBA_AVAILABLE
    _mantel.NUMBA_AVAILABLE = use_numba and old_numba_available
    try:
        start = time.perf_counter()
        result = _mantel._mantel_stats_pearson_flat(x, y_flat, permutations, seed=seed)
        elapsed = time.perf_counter() - start
    finally:
        _mantel.NUMBA_AVAILABLE = old_numba_available
    return result, elapsed


def warm_numba() -> None:
    if not _mantel.NUMBA_AVAILABLE:
        return
    for form in ("full", "condensed"):
        if form == "full":
            x, y_flat = make_full_case(8, 11)
        else:
            x, y_flat = make_condensed_case(8, 11)
        run_once(x, y_flat, permutations=2, seed=22, use_numba=True)


def benchmark_case(form: str, n: int, permutations: int, seed: int) -> None:
    if form == "full":
        x, y_flat = make_full_case(n, seed)
    elif form == "condensed":
        x, y_flat = make_condensed_case(n, seed)
    else:  # pragma: no cover
        raise ValueError(form)

    cy_result, cy_time = run_once(x, y_flat, permutations, seed, use_numba=False)
    nb_result, nb_time = run_once(x, y_flat, permutations, seed, use_numba=True)

    cy_perm = cy_result[2]
    nb_perm = nb_result[2]
    max_diff = float(np.max(np.abs(cy_perm - nb_perm))) if len(cy_perm) else 0.0
    comp_diff = abs(float(cy_result[1]) - float(nb_result[1]))
    print(
        "| Mantel Pearson | {form} | {n} | {perms} | {cy:.6f}s | "
        "{nb:.6f}s | {speedup:.2f}x | {diff:.3e} | comp_diff={comp:.3e} |".format(
            form=form,
            n=n,
            perms=permutations,
            cy=cy_time,
            nb=nb_time,
            speedup=cy_time / nb_time,
            diff=max_diff,
            comp=comp_diff,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--permutations", nargs="+", type=int, default=[999, 9999])
    parser.add_argument(
        "--forms",
        nargs="+",
        choices=("full", "condensed"),
        default=["full", "condensed"],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print_metadata()
    warm_numba()
    print("| Function | Matrix form | n | permutations | Cython | Numba warm | Speedup | Max abs diff | Notes |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for form in args.forms:
        for permutations in args.permutations:
            benchmark_case(form, args.n, permutations, args.seed)


if __name__ == "__main__":
    main()
