#!/usr/bin/env python3
"""
Post-process Schelling-Voter sweep data without loading the
entire sweep into memory.

Expected files
--------------
{data_root}/{regime}/{input_pattern}

Each file is expected to have shape
    (2, 801, 128, 128)
= (species, snapshots, Nx, Ny).

Outputs
-------
1. masses_per_run.csv
   One row per run and stored snapshot, including mean relative entropy.

2. masses_by_parameter.csv
   Mean/std of the spatial masses and mean relative entropy over runs with the same (regime, D_v).

3. observables_spectra_per_run.csv
   One row per run and radial k-bin. Scalar late-time observables are repeated
   across k-bins for convenience.

4. observables_spectra_by_parameter.csv
   Mean/std over runs with the same (regime, D_v), one row per radial k-bin.

5. missing_files.csv
   Only created if files are missing or fail validation.

Notes
-----
- The steady-state window is the last N_AVG snapshots, default 200.
- Power spectra use a 2D orthonormal DCT-II, appropriate for the cell-centered
  finite-volume Neumann Laplacian. Fields are centered snapshot-by-snapshot.
- Radial bins use the exact finite-volume eigen-wavenumber
      k_hat^2 = (2/dx sin(kx dx/2))^2 + (2/dy sin(ky dy/2))^2,
  with DCT mode labels kx = pi*n/Lx and ky = pi*m/Ly.
- The spectrum contains 3 auto-spectra (A, B, vacancy) and the AB cross-spectrum.
- The spectrum is accumulated in chunks to keep memory usage low.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from scipy.fft import dctn


REGIMES = ["segregating", "integrating", "migrating", "well-mixed"]
DV_VALS = np.linspace(0.01, 0.2, 20)
N_RUNS = 10

DEFAULT_L = (50.0, 50.0)
DEFAULT_NUM_BINS = 60
DEFAULT_N_AVG = 200
DEFAULT_DCT_CHUNK = 20
EPS = 1e-10


# ---------------------------------------------------------------------------
# Observables: vectorized versions of the functions in operations.py
# ---------------------------------------------------------------------------

def late_time_observables(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute entropy_index, dissimilarity, and mean_relative_entropy for every
    snapshot in phi.

    Parameters
    ----------
    phi : ndarray, shape (2, T, Nx, Ny)

    Returns
    -------
    H, D, DKL : arrays of shape (T,)
    """
    if phi.ndim != 4 or phi.shape[0] != 2:
        raise ValueError(f"Expected phi shape (2,T,Nx,Ny), got {phi.shape}")

    # Use float64 arithmetic even if data was stored at lower precision.
    A = np.asarray(phi[0], dtype=np.float64)
    B = np.asarray(phi[1], dtype=np.float64)
    phi0 = 1.0 - A - B

    # ---------- dissimilarity(phi) ----------
    gA = A.mean(axis=(1, 2))
    gB = B.mean(axis=(1, 2))
    g0 = phi0.mean(axis=(1, 2))

    gA_safe = np.clip(gA, EPS, None)
    gB_safe = np.clip(gB, EPS, None)
    g0_safe = np.clip(g0, EPS, None)

    D = (
        np.mean(np.abs(A - gA[:, None, None]), axis=(1, 2)) / gA_safe
        + np.mean(np.abs(B - gB[:, None, None]), axis=(1, 2)) / gB_safe
        + np.mean(np.abs(phi0 - g0[:, None, None]), axis=(1, 2)) / g0_safe
    ) / 2.0

    # ---------- mean_relative_entropy(phi) ----------
    global3 = np.stack([gA, gB, g0], axis=1)
    global3 = np.clip(global3, EPS, None)
    Sglobal3 = -np.sum(global3 * np.log(global3), axis=1)

    Aclip = np.clip(A, EPS, None)
    Bclip = np.clip(B, EPS, None)
    Oclip = np.clip(phi0, EPS, None)

    local_kl3 = (
        Aclip * np.log(Aclip / global3[:, 0, None, None])
        + Bclip * np.log(Bclip / global3[:, 1, None, None])
        + Oclip * np.log(Oclip / global3[:, 2, None, None])
    )
    DKL = local_kl3.mean(axis=(1, 2)) / np.clip(Sglobal3, EPS, None)

    # ---------- entropy_index(phi) ----------
    # This preserves the definition in the supplied operations.py:
    # local A/B composition, weighted by local occupancy.
    occ = A + B
    mean_occ = occ.mean(axis=(1, 2))
    global2 = np.stack([gA, gB], axis=1) / np.clip(mean_occ[:, None], EPS, None)
    global2 = np.clip(global2, EPS, None)
    Hglobal2 = -np.sum(global2 * np.log(global2), axis=1)

    pA = np.clip(A / np.clip(occ, EPS, None), EPS, None)
    pB = np.clip(B / np.clip(occ, EPS, None), EPS, None)

    local_kl2 = (
        pA * np.log(pA / global2[:, 0, None, None])
        + pB * np.log(pB / global2[:, 1, None, None])
    )
    occ_sum = occ.sum(axis=(1, 2))
    weights = occ / np.clip(occ_sum[:, None, None], EPS, None)
    H = np.sum(weights * local_kl2, axis=(1, 2)) / np.clip(Hglobal2, EPS, None)

    return H, D, DKL


# ---------------------------------------------------------------------------
# Radially averaged power spectrum
# ---------------------------------------------------------------------------

def make_fv_dct_radial_bins(
    Nx: int,
    Ny: int,
    Lx: float,
    Ly: float,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct radial bins using the eigenvalues of the cell-centered
    second-order finite-volume Neumann Laplacian.

    For a DCT-II mode n = 0,...,N-1,

        k_n = pi*n/L,

    while the corresponding finite-volume eigen-wavenumber is

        k_hat_n = (2/dx) sin(k_n dx/2)
                = (2/dx) sin(pi*n/(2N)).

    The 2D radial coordinate is

        k_hat = sqrt(k_hat_x**2 + k_hat_y**2).

    Returns
    -------
    k_centers : (num_bins-1,)
        Radial finite-volume eigen-wavenumber bin centers.
    bin_index : (Nx*Ny,)
        Flattened radial-bin index for every DCT mode; -1 means excluded.
    mode_count : (num_bins-1,)
        Number of DCT modes in each radial bin.
    """
    dx, dy = Lx / Nx, Ly / Ny

    nx = np.arange(Nx, dtype=np.float64)
    ny = np.arange(Ny, dtype=np.float64)

    # Continuum labels of the Neumann/DCT-II modes.
    kx = np.pi * nx / Lx
    ky = np.pi * ny / Ly

    # Exact eigen-wavenumbers of the second-order FV Neumann Laplacian.
    kx_hat = (2.0 / dx) * np.sin(0.5 * kx * dx)
    ky_hat = (2.0 / dy) * np.sin(0.5 * ky * dy)

    KX_hat, KY_hat = np.meshgrid(kx_hat, ky_hat, indexing="ij")
    k_hat = np.sqrt(KX_hat**2 + KY_hat**2)

    edges = np.linspace(0.0, np.max(k_hat), num_bins)
    centers = 0.5 * (edges[1:] + edges[:-1])

    # Match the old [edge_i, edge_{i+1}) convention.
    idx = np.searchsorted(edges, k_hat.ravel(), side="right") - 1

    # Exclude exactly the largest corner mode, as the previous code excluded
    # k == max(k) through its strict upper-bin comparison.
    valid = (
        (idx >= 0)
        & (idx < len(centers))
        & (k_hat.ravel() < edges[-1])
    )

    bin_dtype = np.int16 if len(centers) < np.iinfo(np.int16).max else np.int32
    bin_index = np.full(k_hat.size, -1, dtype=bin_dtype)
    bin_index[valid] = idx[valid].astype(bin_dtype)

    mode_count = np.bincount(
        bin_index[bin_index >= 0],
        minlength=len(centers),
    ).astype(np.int64)

    return centers, bin_index, mode_count


def radial_average_from_mode_sum(
    mode_sum: np.ndarray,
    bin_index: np.ndarray,
    mode_count: np.ndarray,
    n_snapshots: int,
) -> np.ndarray:
    """Radially average a time-summed DCT quantity."""
    flat = np.asarray(mode_sum, dtype=np.float64).ravel()
    valid = bin_index >= 0
    sums = np.bincount(
        bin_index[valid],
        weights=flat[valid],
        minlength=len(mode_count),
    )
    denom = mode_count.astype(np.float64) * float(n_snapshots)
    out = np.full(len(mode_count), np.nan, dtype=np.float64)
    nonzero = denom > 0
    out[nonzero] = sums[nonzero] / denom[nonzero]
    return out


def power_spectrum_chunked(
    phi: np.ndarray,
    L: tuple[float, float],
    num_bins: int = DEFAULT_NUM_BINS,
    chunk_size: int = DEFAULT_DCT_CHUNK,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the late-time Neumann power spectrum in chunks using a 2D DCT-II.

    The DCT uses ``norm="ortho"`` and the radial coordinate is the exact
    finite-volume eigen-wavenumber k_hat of the cell-centered second-order
    Neumann Laplacian.

    Parameters
    ----------
    phi : ndarray/memmap, shape (2,T,Nx,Ny)
    L : (Lx, Ly)
    num_bins : number of radial bin edges (returns num_bins-1 centers)
    chunk_size : snapshots per DCT batch

    Returns
    -------
    k_centers       : (K,)
        Radial FV eigen-wavenumber k_hat.
    power_spectra   : (3,K)
        DCT auto-spectra for A, B, vacancy.
    G_AB            : (K,)
        DCT AB cross-spectrum.
    """
    if phi.ndim != 4 or phi.shape[0] != 2:
        raise ValueError(f"Expected phi shape (2,T,Nx,Ny), got {phi.shape}")

    _, T, Nx, Ny = phi.shape
    Lx, Ly = L
    k_centers, bin_index, mode_count = make_fv_dct_radial_bins(
        Nx, Ny, Lx, Ly, num_bins
    )

    ps_mode_sum = np.zeros((3, Nx, Ny), dtype=np.float64)
    gab_mode_sum = np.zeros((Nx, Ny), dtype=np.float64)

    for start in range(0, T, chunk_size):
        stop = min(start + chunk_size, T)

        # Copy only this small chunk from memmap and use time-first layout.
        A = np.array(phi[0, start:stop], dtype=np.float64, copy=True)
        B = np.array(phi[1, start:stop], dtype=np.float64, copy=True)
        O = 1.0 - A - B

        # Remove the spatially constant DCT mode independently per snapshot.
        A -= A.mean(axis=(1, 2), keepdims=True)
        B -= B.mean(axis=(1, 2), keepdims=True)
        O -= O.mean(axis=(1, 2), keepdims=True)

        # DCT-II diagonalizes the cell-centered FV Neumann Laplacian.
        A_k = dctn(A, type=2, axes=(1, 2), norm="ortho")
        B_k = dctn(B, type=2, axes=(1, 2), norm="ortho")
        O_k = dctn(O, type=2, axes=(1, 2), norm="ortho")

        # DCT coefficients are real for real input fields.
        ps_mode_sum[0] += np.sum(A_k**2, axis=0)
        ps_mode_sum[1] += np.sum(B_k**2, axis=0)
        ps_mode_sum[2] += np.sum(O_k**2, axis=0)
        gab_mode_sum += np.sum(A_k * B_k, axis=0)

    ps = np.vstack([
        radial_average_from_mode_sum(ps_mode_sum[a], bin_index, mode_count, T)
        for a in range(3)
    ])
    gab = radial_average_from_mode_sum(gab_mode_sum, bin_index, mode_count, T)

    return k_centers, ps, gab


def spectral_kmean(power_spectrum: np.ndarray, k_bins: np.ndarray) -> float:
    """Same convention as the previous kmean helper."""
    Z = np.nansum(power_spectrum)
    if not np.isfinite(Z) or Z <= 0:
        return np.nan
    return float(np.nansum(k_bins * power_spectrum) / Z)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def write_mass_grouped_csv(
    path: Path,
    mass_sum: np.ndarray,
    mass_sumsq: np.ndarray,
    mass_count: np.ndarray,
    regimes: list[str],
    dv_vals: np.ndarray,
):
    """
    Arrays have shape:
      sum/sumsq: (R, Dv, T, 5) for rhoA, rhoB, rho0, rho_occ, DKL
      count:     (R, Dv)
    """
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "regime", "D_v", "snapshot", "n_runs",
            "rhoA_mean", "rhoA_std",
            "rhoB_mean", "rhoB_std",
            "rho0_mean", "rho0_std",
            "rho_occ_mean", "rho_occ_std",
            "DKL_mean", "DKL_std",
        ])

        for r, regime in enumerate(regimes):
            for i, dv in enumerate(dv_vals):
                n = int(mass_count[r, i])
                if n == 0:
                    continue

                mean = mass_sum[r, i] / n
                var = np.maximum(mass_sumsq[r, i] / n - mean**2, 0.0)
                std = np.sqrt(var)

                for t in range(mean.shape[0]):
                    w.writerow([
                        regime, f"{dv:.3f}", t, n,
                        mean[t, 0], std[t, 0],
                        mean[t, 1], std[t, 1],
                        mean[t, 2], std[t, 2],
                        mean[t, 3], std[t, 3],
                        mean[t, 4], std[t, 4],
                    ])


def mean_std(sum_: np.ndarray, sumsq: np.ndarray, n: int):
    if n == 0:
        return np.full_like(sum_, np.nan), np.full_like(sum_, np.nan)
    mean = sum_ / n
    var = np.maximum(sumsq / n - mean**2, 0.0)
    return mean, np.sqrt(var)



def format_input_filename(pattern: str, regime: str, n_run: int, dv: float) -> str:
    """Format a user-supplied input filename pattern.

    Available fields are {regime}, {n_run}, {D_v}, {Dv}, and {dv}.
    The three diffusion aliases all refer to the same floating-point value.
    Example: VS_run_{n_run}_det_Dv{D_v:.3f}.npy
    """
    return pattern.format(
        regime=regime,
        n_run=n_run,
        D_v=dv,
        Dv=dv,
        dv=dv,
    )

# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process(args):
    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    regimes = REGIMES
    dv_vals = DV_VALS
    n_runs = N_RUNS
    L = (args.Lx, args.Ly)

    # We discover T and spatial shape from the first valid file.
    sample_shape = None
    for regime in regimes:
        for dv in dv_vals:
            for n_run in range(n_runs):
                p = data_root / regime / format_input_filename(args.input_pattern, regime, n_run, dv)
                if p.exists():
                    arr = np.load(p, mmap_mode="r")
                    sample_shape = arr.shape
                    del arr
                    break
            if sample_shape is not None:
                break
        if sample_shape is not None:
            break

    if sample_shape is None:
        raise FileNotFoundError(
            f"No sweep files found below {data_root.resolve()}"
        )

    if len(sample_shape) != 4 or sample_shape[0] != 2:
        raise ValueError(
            f"Expected files shaped (2,T,Nx,Ny), first file has {sample_shape}"
        )

    _, T, Nx, Ny = sample_shape
    n_avg = min(args.n_avg, T)

    print(f"Detected data shape: {sample_shape}")
    print(f"Late-time averaging window: last {n_avg} snapshots")
    print(f"Domain: L=({args.Lx}, {args.Ly}), grid=({Nx}, {Ny})")
    print(f"Expected sweep size: {len(regimes) * len(dv_vals) * n_runs} files")

    # Snapshot accumulators: rhoA, rhoB, rho0, rho_occ, DKL.
    mass_sum = np.zeros((len(regimes), len(dv_vals), T, 5), dtype=np.float64)
    mass_sumsq = np.zeros_like(mass_sum)
    mass_count = np.zeros((len(regimes), len(dv_vals)), dtype=np.int32)

    # Scalar grouped observables in order:
    # H_mean, D_mean, DKL_mean, H_temporal_std, D_temporal_std,
    # DKL_temporal_std, kmean, kpeak_A
    n_scalar = 8
    scalar_sum = np.zeros((len(regimes), len(dv_vals), n_scalar), dtype=np.float64)
    scalar_sumsq = np.zeros_like(scalar_sum)

    # Spectrum accumulators are initialized after first valid spectrum.
    spec_sum = None
    spec_sumsq = None
    gab_sum = None
    gab_sumsq = None
    k_reference = None
    obs_count = np.zeros((len(regimes), len(dv_vals)), dtype=np.int32)

    missing = []

    masses_path = out_dir / "masses_per_run.csv"
    obs_path = out_dir / "observables_spectra_per_run.csv"

    with masses_path.open("w", newline="") as fm, obs_path.open("w", newline="") as fo:
        wm = csv.writer(fm)
        wo = csv.writer(fo)

        wm.writerow([
            "regime", "D_v", "n_run", "snapshot",
            "rhoA", "rhoB", "rho0", "rho_occ", "DKL"
        ])

        wo.writerow([
            "regime", "D_v", "n_run", "n_steady_snapshots",
            "H_mean", "H_std_time",
            "D_mean", "D_std_time",
            "DKL_mean", "DKL_std_time",
            "kmean_AB", "kpeak_A",
            "k_bin", "k",
            "PS_A", "PS_B", "PS_0", "G_AB",
        ])

        completed = 0
        total = len(regimes) * len(dv_vals) * n_runs

        for r, regime in enumerate(regimes):
            for i, dv in enumerate(dv_vals):
                for n_run in range(n_runs):
                    path = data_root / regime / format_input_filename(args.input_pattern, regime, n_run, dv)

                    if not path.exists():
                        msg = f"Missing: {path}"
                        print(msg, file=sys.stderr)
                        missing.append([regime, f"{dv:.3f}", n_run, str(path), "missing"])
                        if args.strict:
                            raise FileNotFoundError(path)
                        continue

                    try:
                        phi = np.load(path, mmap_mode="r")
                        if phi.shape != sample_shape:
                            raise ValueError(
                                f"shape {phi.shape} != expected {sample_shape}"
                            )

                        # ---------------------------------------------------
                        # 1) Spatial probability masses over all snapshots.
                        # ---------------------------------------------------
                        masses = np.empty((T, 5), dtype=np.float64)
                        for start in range(0, T, args.mass_chunk):
                            stop = min(start + args.mass_chunk, T)
                            A = np.asarray(phi[0, start:stop], dtype=np.float64)
                            B = np.asarray(phi[1, start:stop], dtype=np.float64)
                            mA = A.mean(axis=(1, 2))
                            mB = B.mean(axis=(1, 2))
                            masses[start:stop, 0] = mA
                            masses[start:stop, 1] = mB
                            masses[start:stop, 2] = 1.0 - mA - mB
                            masses[start:stop, 3] = mA + mB

                            # Match fhd.mean_relative_entropy / operations.mean_relative_entropy
                            # for every stored snapshot.
                            _, _, dkl_chunk = late_time_observables(
                                np.stack([A, B], axis=0)
                            )
                            masses[start:stop, 4] = dkl_chunk

                        for t in range(T):
                            wm.writerow([
                                regime, f"{dv:.3f}", n_run, t,
                                masses[t, 0], masses[t, 1], masses[t, 2],
                                masses[t, 3], masses[t, 4],
                            ])

                        mass_sum[r, i] += masses
                        mass_sumsq[r, i] += masses**2
                        mass_count[r, i] += 1

                        # ---------------------------------------------------
                        # 2) Late-time scalar observables.
                        # ---------------------------------------------------
                        tail = np.asarray(phi[:, -n_avg:], dtype=np.float64)
                        H_t, D_t, DKL_t = late_time_observables(tail)

                        H_mean, H_std_t = float(np.mean(H_t)), float(np.std(H_t))
                        D_mean, D_std_t = float(np.mean(D_t)), float(np.std(D_t))
                        DKL_mean, DKL_std_t = float(np.mean(DKL_t)), float(np.std(DKL_t))

                        # ---------------------------------------------------
                        # 3) Late-time power spectrum.
                        # ---------------------------------------------------
                        k, ps, gab = power_spectrum_chunked(
                            tail,
                            L=L,
                            num_bins=args.num_bins,
                            chunk_size=args.fft_chunk,
                        )

                        if k_reference is None:
                            k_reference = k.copy()
                            K = len(k_reference)
                            spec_sum = np.zeros(
                                (len(regimes), len(dv_vals), 3, K),
                                dtype=np.float64
                            )
                            spec_sumsq = np.zeros_like(spec_sum)
                            gab_sum = np.zeros(
                                (len(regimes), len(dv_vals), K),
                                dtype=np.float64
                            )
                            gab_sumsq = np.zeros_like(gab_sum)
                        elif not np.allclose(k, k_reference, rtol=0, atol=1e-12):
                            raise ValueError("k bins changed between files")

                        kmean_ab = spectral_kmean(ps[0] + ps[1], k)
                        if np.all(np.isnan(ps[0])):
                            kpeak_A = np.nan
                        else:
                            kpeak_A = float(k[np.nanargmax(ps[0])])

                        scalar = np.array([
                            H_mean, D_mean, DKL_mean,
                            H_std_t, D_std_t, DKL_std_t,
                            kmean_ab, kpeak_A,
                        ], dtype=np.float64)

                        scalar_sum[r, i] += scalar
                        scalar_sumsq[r, i] += scalar**2
                        spec_sum[r, i] += ps
                        spec_sumsq[r, i] += ps**2
                        gab_sum[r, i] += gab
                        gab_sumsq[r, i] += gab**2
                        obs_count[r, i] += 1

                        for kb, kval in enumerate(k):
                            wo.writerow([
                                regime, f"{dv:.3f}", n_run, n_avg,
                                H_mean, H_std_t,
                                D_mean, D_std_t,
                                DKL_mean, DKL_std_t,
                                kmean_ab, kpeak_A,
                                kb, kval,
                                ps[0, kb], ps[1, kb], ps[2, kb], gab[kb],
                            ])

                        del tail, phi

                    except Exception as exc:
                        print(f"Failed: {path}: {exc}", file=sys.stderr)
                        missing.append([
                            regime, f"{dv:.3f}", n_run, str(path),
                            f"{type(exc).__name__}: {exc}"
                        ])
                        if args.strict:
                            raise

                    completed += 1
                    if completed % args.report_every == 0 or completed == total:
                        print(f"Processed sweep entries: {completed}/{total}")

    # -----------------------------------------------------------------------
    # Grouped masses over runs with same regime and D_v.
    # -----------------------------------------------------------------------
    masses_grouped_path = out_dir / "masses_by_parameter.csv"
    write_mass_grouped_csv(
        masses_grouped_path,
        mass_sum,
        mass_sumsq,
        mass_count,
        regimes,
        dv_vals,
    )

    # -----------------------------------------------------------------------
    # Grouped observables + spectra.
    # -----------------------------------------------------------------------
    obs_grouped_path = out_dir / "observables_spectra_by_parameter.csv"
    with obs_grouped_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "regime", "D_v", "n_runs", "n_steady_snapshots",
            "H_mean_runs", "H_std_runs",
            "D_mean_runs", "D_std_runs",
            "DKL_mean_runs", "DKL_std_runs",
            "H_time_std_mean_runs",
            "D_time_std_mean_runs",
            "DKL_time_std_mean_runs",
            "kmean_AB_mean_runs", "kmean_AB_std_runs",
            "kpeak_A_mean_runs", "kpeak_A_std_runs",
            "k_bin", "k",
            "PS_A_mean", "PS_A_std",
            "PS_B_mean", "PS_B_std",
            "PS_0_mean", "PS_0_std",
            "G_AB_mean", "G_AB_std",
        ])

        if k_reference is not None:
            for r, regime in enumerate(regimes):
                for i, dv in enumerate(dv_vals):
                    n = int(obs_count[r, i])
                    if n == 0:
                        continue

                    sm, ss = mean_std(scalar_sum[r, i], scalar_sumsq[r, i], n)
                    pm, psd = mean_std(spec_sum[r, i], spec_sumsq[r, i], n)
                    gm, gsd = mean_std(gab_sum[r, i], gab_sumsq[r, i], n)

                    for kb, kval in enumerate(k_reference):
                        w.writerow([
                            regime, f"{dv:.3f}", n, n_avg,
                            sm[0], ss[0],
                            sm[1], ss[1],
                            sm[2], ss[2],
                            sm[3], sm[4], sm[5],
                            sm[6], ss[6],
                            sm[7], ss[7],
                            kb, kval,
                            pm[0, kb], psd[0, kb],
                            pm[1, kb], psd[1, kb],
                            pm[2, kb], psd[2, kb],
                            gm[kb], gsd[kb],
                        ])

    # Missing/failed file report.
    if missing:
        missing_path = out_dir / "missing_files.csv"
        with missing_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["regime", "D_v", "n_run", "path", "status"])
            w.writerows(missing)
        print(f"Missing/failed entries written to: {missing_path}")

    print("\nDone. Wrote:")
    print(f"  {masses_path}")
    print(f"  {masses_grouped_path}")
    print(f"  {obs_path}")
    print(f"  {obs_grouped_path}")
    print(f"Successful parameter/run files: {int(obs_count.sum())}")
    print(f"Missing/failed files: {len(missing)}")


def build_parser():
    p = argparse.ArgumentParser(
        description="Post-process Schelling-Voter sweep data."
    )
    p.add_argument(
        "--data-root", default="data",
        help="Root directory containing regime subdirectories (default: data)"
    )
    p.add_argument(
        "--input-pattern",
        default="VS_run_{n_run}_det_Dv{D_v:.3f}.npy",
        help=(
            "Filename pattern inside each regime directory. Available fields: "
            "{regime}, {n_run}, {D_v}, {Dv}, {dv}. "
            "Default: VS_run_{n_run}_det_Dv{D_v:.3f}.npy"
        ),
    )
    p.add_argument(
        "--output-dir", default="processed",
        help="Directory for CSV outputs (default: processed)"
    )
    p.add_argument(
        "--n-avg", type=int, default=DEFAULT_N_AVG,
        help="Number of final snapshots for observables/spectra (default: 200)"
    )
    p.add_argument(
        "--num-bins", type=int, default=DEFAULT_NUM_BINS,
        help="Number of radial k-bin edges; output has num_bins-1 bins (default: 60)"
    )
    p.add_argument(
        "--Lx", type=float, default=DEFAULT_L[0],
        help="Physical x-length (default: 50)"
    )
    p.add_argument(
        "--Ly", type=float, default=DEFAULT_L[1],
        help="Physical y-length (default: 50)"
    )
    p.add_argument(
        "--dct-chunk", "--fft-chunk",
        dest="fft_chunk",
        type=int,
        default=DEFAULT_DCT_CHUNK,
        help=(
            "Snapshots per DCT batch (default: 20). "
            "--fft-chunk is retained as a backwards-compatible alias."
        ),
    )
    p.add_argument(
        "--mass-chunk", type=int, default=64,
        help="Snapshots per mass-computation batch (default: 64)"
    )
    p.add_argument(
        "--report-every", type=int, default=10,
        help="Print progress every N expected files (default: 10)"
    )
    p.add_argument(
        "--strict", action="store_true",
        help="Abort on the first missing or invalid file"
    )
    return p


if __name__ == "__main__":
    parser = build_parser()
    process(parser.parse_args())
