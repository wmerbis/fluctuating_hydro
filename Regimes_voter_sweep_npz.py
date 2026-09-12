#!/usr/bin/env python3
"""
Cluster parameter sweep for the fluctuating Schelling--Voter model.

Runs 4 regimes x 10 realizations x 20 D_v values = 800 simulations.
Each simulation writes ONE compressed NPZ file containing reduced observables,
late-time summaries, the final field, projection diagnostics, and reproduction
metadata. The full phi_run is used only in memory and is NOT written to disk.

The SLURM array/chunk logic is intentionally kept compatible with the previous
Regimes_voter_sweep.py workflow via SLURM_ARRAY_TASK_ID and
PARAMETER_CHUNK_SIZE.
"""
from __future__ import annotations

import gc
import json
import multiprocessing
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import scipy
from scipy.fft import dctn

try:
    import fhd
except ImportError:
    import fhd.FHD_2D as fhd

try:
    from fhd.operations import dissimilarity, mean_relative_entropy
except ImportError:
    from operations import dissimilarity, mean_relative_entropy


# =============================================================================
# Sweep / simulation configuration
# =============================================================================

N = (128, 128)
L = (50.0, 50.0)
DT = 1.0e-3
NSTEPS = 2_000_000
NFRAMES = 1000
SAVE_EVERY = NSTEPS // NFRAMES

if NSTEPS % NFRAMES != 0:
    raise ValueError("NSTEPS must be divisible by NFRAMES")

REGIMES = ("segregating", "integrating", "migrating", "well-mixed")
KAPPAS = {
    "segregating": np.array([[0.6, -0.4], [-0.4, 0.6]], dtype=float),
    "integrating": np.array([[0.6, 1.0], [1.0, 0.6]], dtype=float),
    "migrating": np.array([[1.0, 1.0], [-1.0, 1.0]], dtype=float),
    "well-mixed": np.zeros((2, 2), dtype=float),
}

D = np.array([0.1, 0.1], dtype=float)
GAMMA = np.eye(2, dtype=float)
DV_VALS = np.linspace(0.01, 0.20, 20)
N_RUNS = 10
BETA = 10.0
H = 0.01
NOISE_V = 1.0
NOISE = True

BC = "Neumann"
SCHELLING_FLUX = "finite_volume"
PROJECTION_MODE = "redistribute"
USE_NUMBA_PROJECTION = True
NUMBA_PROJECTION_THREADS = 1
VOTER_NOISE_MODE = "wright_fisher"
WF_GAUSSIAN_THRESHOLD = 0.01
REACTION_DIAGNOSTICS = False
USE_FASTPATH = True
SCHEME = "FE"
MODEL = "Schelling+Voter"

# Preserve the original sweep's initial-condition scale.
RHO_A0 = 0.35
RHO_B0 = 0.35
INITIAL_PERTURBATION = 0.05

# Reduced-output settings.
N_SPECTRUM_BINS = 50
TAIL_FRAMES = 200
SPECTRUM_USE_LATTICE_K = True
SPECTRUM_CENTERED = True
TAIL_STD_DDOF = 1

OUTPUT_ROOT = Path(os.environ.get("SWEEP_OUTPUT_DIR", "data"))
RESUME = True

# Seed policy:
# - same initial condition for a given (regime, run) across all D_v values;
# - independent simulation-noise seed for every one of the 800 parameter points.
INIT_SEED_BASE = 31_000_000
SIM_SEED_BASE = 71_000_000


# =============================================================================
# Reproduction metadata helpers
# =============================================================================

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _fhd_version() -> str:
    return str(getattr(fhd, "__version__", "unknown"))


def _json_scalar(obj) -> np.ndarray:
    """Store a JSON object inside NPZ without requiring allow_pickle=True."""
    return np.array(json.dumps(obj, sort_keys=True))


def run_path(regime: str, run_index: int, dv_index: int, D_v: float) -> Path:
    out_dir = OUTPUT_ROOT / regime
    return out_dir / f"VS_run_{run_index:02d}_Dv{dv_index:02d}_{D_v:.3f}.npz"


def _complete_npz(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            required = {
                "metadata_json",
                "time",
                "dissimilarity",
                "S_pol",
                "k_mean_pol",
                "interface_density",
                "phi_final",
            }
            return required.issubset(z.files)
    except Exception:
        return False


def atomic_savez_compressed(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


# =============================================================================
# Initial condition / simulator
# =============================================================================

def make_initial_condition(seed: int) -> tuple[np.ndarray, int, int]:
    """Gaussian initial state, with only unphysical cells redrawn.

    The original sweep used sigma=0.05 independently for A and B.  At 128^2
    cells that occasionally gives A+B>1 in a few cells.  Redrawing only those
    cells preserves the intended perturbation scale while ensuring the solver
    starts inside the physical simplex, so projection diagnostics are not
    contaminated by an invalid input field.
    """
    rng = np.random.default_rng(seed)
    A = RHO_A0 + INITIAL_PERTURBATION * rng.normal(size=N)
    B = RHO_B0 + INITIAL_PERTURBATION * rng.normal(size=N)

    bad = (A < 0.0) | (B < 0.0) | (A + B > 1.0)
    first_pass_bad = int(np.count_nonzero(bad))
    redraws_total = 0

    while np.any(bad):
        nbad = int(np.count_nonzero(bad))
        redraws_total += nbad
        A[bad] = RHO_A0 + INITIAL_PERTURBATION * rng.normal(size=nbad)
        B[bad] = RHO_B0 + INITIAL_PERTURBATION * rng.normal(size=nbad)
        bad = (A < 0.0) | (B < 0.0) | (A + B > 1.0)

    phi = np.stack((A, B), axis=0).astype(np.float64, copy=False)
    return phi, first_pass_bad, redraws_total


def make_param(regime: str, D_v: float) -> dict:
    return {
        "D": D.copy(),
        "Gamma": GAMMA.copy(),
        "D_v": float(D_v),
        "beta": float(BETA),
        "h": float(H),
        "noise_v": float(NOISE_V),
        "kappa": KAPPAS[regime].copy(),
    }


def make_simulator():
    sim = fhd.fhd_2d(
        L,
        N,
        bc=BC,
        fft=False,
        schelling_flux=SCHELLING_FLUX,
        projection_mode=PROJECTION_MODE,
        use_numba_projection=USE_NUMBA_PROJECTION,
        numba_projection_threads=NUMBA_PROJECTION_THREADS,
        voter_noise_mode=VOTER_NOISE_MODE,
        wf_gaussian_threshold=WF_GAUSSIAN_THRESHOLD,
        reaction_diagnostic=REACTION_DIAGNOSTICS,
    )

    # Keep reaction diagnostics off even on development versions exposing a
    # separate attribute spelling.
    for name in (
        "reaction_diagnostic",
        "reaction_diagnostics",
        "schelling_reaction_diagnostics",
        "enable_reaction_diagnostics",
    ):
        if hasattr(sim, name):
            setattr(sim, name, False)
    return sim


# =============================================================================
# DCT-II spectra with the finite-volume lattice wavenumber
# =============================================================================

def make_spectrum_geometry(N, L, num_bins: int):
    """
    Geometry matching operations.power_spectrum_dct_2d(..., use_lattice_k=True).

    DCT-II Neumann continuum modes are k_n = pi*n/L.  For the finite-volume
    Laplacian we bin by the exact lattice wavenumber
        khat_i = (2/dx_i) sin(k_i dx_i / 2).
    """
    Nx, Ny = N
    Lx, Ly = L
    dx, dy = Lx / Nx, Ly / Ny

    kx = np.pi * np.arange(Nx, dtype=float) / Lx
    ky = np.pi * np.arange(Ny, dtype=float) / Ly

    if SPECTRUM_USE_LATTICE_K:
        kx = (2.0 / dx) * np.sin(0.5 * kx * dx)
        ky = (2.0 / dy) * np.sin(0.5 * ky * dy)

    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k2d = np.sqrt(KX * KX + KY * KY)
    kflat = k2d.ravel()

    edges = np.linspace(0.0, float(kflat.max()), num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Precompute bin membership once. searchsorted(..., side='right') reproduces
    # [edge_i, edge_{i+1}) with the upper endpoint included in the final bin.
    bin_index = np.searchsorted(edges, kflat, side="right") - 1
    bin_index[kflat == edges[-1]] = num_bins - 1
    valid = (bin_index >= 0) & (bin_index < num_bins)
    counts = np.bincount(bin_index[valid], minlength=num_bins).astype(np.int64)

    return k2d, centers, edges, bin_index, valid, counts


K2D, K_CENTERS, K_EDGES, K_BIN_INDEX, K_BIN_VALID, K_COUNTS = make_spectrum_geometry(
    N, L, N_SPECTRUM_BINS
)

SPECTRUM_NAMES = (
    "S_AA",
    "S_BB",
    "S_00",
    "S_AB",
    "S_A0",
    "S_B0",
    "S_pol",
    "S_occ",
)


def _radial_average(power2d: np.ndarray) -> np.ndarray:
    p = np.asarray(power2d, dtype=float).ravel()
    sums = np.bincount(
        K_BIN_INDEX[K_BIN_VALID],
        weights=p[K_BIN_VALID],
        minlength=N_SPECTRUM_BINS,
    )
    out = np.full(N_SPECTRUM_BINS, np.nan, dtype=float)
    nonzero = K_COUNTS > 0
    out[nonzero] = sums[nonzero] / K_COUNTS[nonzero]
    return out


def frame_dct_spectra(phi: np.ndarray):
    """
    Return radially averaged DCT-II spectra for all independent A/B/vacancy
    pairs plus polarization/occupancy, and exact <k> of polarization.

    Normalization follows operations.power_spectrum_dct_2d: orthonormal DCT-II
    coefficients are squared/cross-multiplied without an additional Ncell
    factor.  <k> is evaluated from the full 2-D modal polarization power.
    """
    A, B = phi
    vac = 1.0 - A - B
    fields = np.stack((A, B, vac), axis=0).astype(float, copy=False)

    if SPECTRUM_CENTERED:
        fields = fields - fields.mean(axis=(1, 2), keepdims=True)

    coeff = dctn(fields, type=2, axes=(1, 2), norm="ortho")
    Ah, Bh, Vh = coeff

    P_AA = Ah * Ah
    P_BB = Bh * Bh
    P_00 = Vh * Vh
    P_AB = Ah * Bh
    P_A0 = Ah * Vh
    P_B0 = Bh * Vh
    P_pol = (Ah - Bh) ** 2
    P_occ = (Ah + Bh) ** 2

    # Remove the zero mode explicitly before treating polarization power as a
    # probability distribution over k. Centering already makes this ~0, but
    # the explicit assignment avoids roundoff entering the denominator.
    P_pol[0, 0] = 0.0
    denom = float(P_pol.sum())
    if denom > 0.0:
        k_mean = float(np.sum(K2D * P_pol) / denom)
    else:
        k_mean = np.nan

    spectra = (
        P_AA,
        P_BB,
        P_00,
        P_AB,
        P_A0,
        P_B0,
        P_pol,
        P_occ,
    )
    radial = {name: _radial_average(power) for name, power in zip(SPECTRUM_NAMES, spectra)}
    return radial, k_mean


# =============================================================================
# Scalar observables / interface density
# =============================================================================

def interface_density_frame(phi: np.ndarray, eps: float = 1.0e-14) -> float:
    """
    Voter interface density 1 - <m_i m_j> over nearest-neighbor bonds.

    For these Neumann runs, only physical interior bonds are counted.  This is
    deliberately different from operations.voter_interface_density(), whose
    np.roll implementation wraps the lattice and is therefore periodic.
    """
    A, B = phi
    occ = A + B
    m = np.divide(A - B, occ, out=np.zeros_like(A), where=occ > eps)

    if BC.lower() == "periodic":
        corr_x = np.mean(m * np.roll(m, -1, axis=0))
        corr_y = np.mean(m * np.roll(m, -1, axis=1))
        corr = 0.5 * (corr_x + corr_y)
    elif BC.lower() == "neumann":
        sx = np.sum(m[:-1, :] * m[1:, :])
        sy = np.sum(m[:, :-1] * m[:, 1:])
        n_bonds = (m.shape[0] - 1) * m.shape[1] + m.shape[0] * (m.shape[1] - 1)
        corr = (sx + sy) / n_bonds
    else:
        raise ValueError(f"Unsupported BC for interface density: {BC}")

    return float(1.0 - corr)


def projection_arrays(sim, dt: float) -> tuple[dict[str, np.ndarray], dict]:
    """Extract cumulative scalar projection diagnostics and optional history."""
    work = getattr(sim, "_work", None)
    if not isinstance(work, dict):
        return {}, {}

    diag = work.get("projection_diag", {})
    if not isinstance(diag, dict):
        return {}, {}

    history = diag.get("history", [])
    summary = {}
    arrays = {}

    for key, value in diag.items():
        if key == "history":
            continue
        if isinstance(value, (bool, int, float, np.integer, np.floating)):
            scalar = value.item() if isinstance(value, np.generic) else value
            summary[key] = scalar
            arrays[f"proj_{key}"] = np.asarray(scalar)

    # Same bookkeeping checks used in the original sweep helper.
    expected_net = (
        float(summary.get("mass_added_low", 0.0))
        - float(summary.get("mass_removed_high", 0.0))
        - float(summary.get("mass_removed_simplex", 0.0))
        + float(summary.get("mass_added_transfer_fallback", 0.0))
        + float(summary.get("mass_roundoff_cleanup", 0.0))
    )
    summary["expected_net_change"] = expected_net
    summary["bookkeeping_error"] = float(summary.get("net_mass_change_projection", 0.0)) - expected_net
    arrays["proj_expected_net_change"] = np.asarray(expected_net)
    arrays["proj_bookkeeping_error"] = np.asarray(summary["bookkeeping_error"])

    if history:
        arrays["proj_hist_step"] = np.array([h.get("step", -1) for h in history], dtype=np.int64)
        arrays["proj_hist_time"] = arrays["proj_hist_step"].astype(float) * dt
        arrays["proj_hist_stage"] = np.array(
            [str(h.get("stage", "unknown")) for h in history], dtype="U32"
        )

        # Save every numeric history field encountered, not just a hard-coded
        # subset, so future projection diagnostics remain available.
        numeric_fields = sorted(
            {
                key
                for row in history
                if isinstance(row, dict)
                for key, value in row.items()
                if key not in {"step", "stage"}
                and isinstance(value, (bool, int, float, np.integer, np.floating))
            }
        )
        for field in numeric_fields:
            arrays[f"proj_hist_{field}"] = np.array(
                [row.get(field, np.nan) for row in history], dtype=float
            )

    return arrays, summary


def extract_reduced_output(phi_run: np.ndarray) -> dict[str, np.ndarray]:
    if phi_run.ndim != 4 or phi_run.shape[0] != 2 or tuple(phi_run.shape[2:]) != tuple(N):
        raise ValueError(f"Unexpected phi_run shape: {phi_run.shape}")

    nf = phi_run.shape[1]
    expected_nf = NFRAMES + 1  # fhd.run includes the initial condition at t=0

    if nf != expected_nf:
        raise ValueError(
            f"Expected {expected_nf} stored frames "
            f"(initial condition + {NFRAMES} recorded frames), got {nf}"
        )
    out: dict[str, np.ndarray] = {
        "frame_index": np.arange(nf, dtype=np.int64),
        "frame_step": np.arange(nf, dtype=np.int64) * SAVE_EVERY,
        "time": np.arange(nf, dtype=float) * (SAVE_EVERY * DT),
        "mean_A": np.empty(nf),
        "mean_B": np.empty(nf),
        "mean_0": np.empty(nf),
        "mean_pol": np.empty(nf),
        "mean_occ": np.empty(nf),
        "var_A": np.empty(nf),
        "var_B": np.empty(nf),
        "var_0": np.empty(nf),
        "var_pol": np.empty(nf),
        "var_occ": np.empty(nf),
        "dissimilarity": np.empty(nf),
        "relative_entropy": np.empty(nf),
        "k_mean_pol": np.empty(nf),
        "interface_density": np.empty(nf),
        "k_centers": K_CENTERS.copy(),
        "k_edges": K_EDGES.copy(),
        "spectrum_counts": K_COUNTS.copy(),
        "spectrum_use_lattice_k": np.asarray(SPECTRUM_USE_LATTICE_K),
        "spectrum_dct_type": np.asarray(2),
        "spectrum_centered": np.asarray(SPECTRUM_CENTERED),
    }

    for name in SPECTRUM_NAMES:
        out[name] = np.empty((nf, N_SPECTRUM_BINS), dtype=float)

    # Online late-time field statistics avoid allocating another 200-frame copy.
    tail_start = max(0, nf - TAIL_FRAMES)
    tail_n = nf - tail_start
    tail_count = 0
    phi_tail_mean = np.zeros((2,) + N, dtype=np.float64)
    phi_tail_M2 = np.zeros((2,) + N, dtype=np.float64)

    for j in range(nf):
        phi = phi_run[:, j]
        A, B = phi
        vac = 1.0 - A - B
        pol = A - B
        occ = A + B

        out["mean_A"][j] = A.mean()
        out["mean_B"][j] = B.mean()
        out["mean_0"][j] = vac.mean()
        out["mean_pol"][j] = pol.mean()
        out["mean_occ"][j] = occ.mean()

        out["var_A"][j] = A.var()
        out["var_B"][j] = B.var()
        out["var_0"][j] = vac.var()
        out["var_pol"][j] = pol.var()
        out["var_occ"][j] = occ.var()

        out["dissimilarity"][j] = dissimilarity(phi)
        out["relative_entropy"][j] = mean_relative_entropy(phi)
        out["interface_density"][j] = interface_density_frame(phi)

        spectra, k_mean = frame_dct_spectra(phi)
        out["k_mean_pol"][j] = k_mean
        for name, values in spectra.items():
            out[name][j] = values

        if j >= tail_start:
            # Welford update for numerically stable per-cell temporal mean/std.
            tail_count += 1
            delta = phi - phi_tail_mean
            phi_tail_mean += delta / tail_count
            phi_tail_M2 += delta * (phi - phi_tail_mean)

    # Exact final configuration, but not the full trajectory.
    out["phi_final"] = phi_run[:, -1].copy()

    # Last-200-frame scalar summaries.
    tail = slice(tail_start, nf)
    out["tail_frame_indices"] = np.arange(tail_start, nf, dtype=np.int64)
    out["tail_time_start"] = np.asarray(out["time"][tail_start])
    out["tail_nframes"] = np.asarray(tail_n)

    for key in ("dissimilarity", "relative_entropy", "k_mean_pol", "interface_density"):
        vals = out[key][tail]
        out[f"{key}_tail_mean"] = np.asarray(np.nanmean(vals))
        out[f"{key}_tail_std"] = np.asarray(np.nanstd(vals, ddof=TAIL_STD_DDOF))

    for name in SPECTRUM_NAMES:
        vals = out[name][tail]
        with np.errstate(invalid="ignore"):
            out[f"{name}_tail_mean"] = np.nanmean(vals, axis=0)
            out[f"{name}_tail_std"] = np.nanstd(vals, axis=0, ddof=TAIL_STD_DDOF)

    # Per-cell temporal mean/std over the same final 200 frames. This is cheap
    # compared with storing those 200 complete configurations and is useful for
    # identifying persistent vs fluctuating spatial structure.
    if tail_count != tail_n:
        raise RuntimeError(f"Tail-statistics bookkeeping mismatch: {tail_count} != {tail_n}")
    if tail_n > 1:
        phi_tail_std = np.sqrt(np.maximum(phi_tail_M2, 0.0) / (tail_n - 1))
    else:
        phi_tail_std = np.full_like(phi_tail_mean, np.nan)
    out["phi_tail_mean"] = phi_tail_mean
    out["phi_tail_std"] = phi_tail_std

    return out


# =============================================================================
# One simulation
# =============================================================================

def run_simulation(param_set):
    task_id, regime_index, regime, run_index, dv_index, D_v = param_set
    pid = os.getpid()
    path = run_path(regime, run_index, dv_index, D_v)

    try:
        if RESUME and _complete_npz(path):
            print(f"[pid={pid}] SKIP existing complete run: {path}", flush=True)
            return

        init_seed = INIT_SEED_BASE + regime_index * N_RUNS + run_index
        sim_seed = SIM_SEED_BASE + task_id

        phi_init, init_bad_cells, init_redraws = make_initial_condition(init_seed)
        param = make_param(regime, D_v)
        sim = make_simulator()

        sim.set_seed(sim_seed)
        # Some legacy Gaussian paths still use numpy's global RNG.
        np.random.seed(sim_seed)

        print(
            f"[pid={pid}] START task={task_id:03d} regime={regime} "
            f"run={run_index:02d} D_v={D_v:.3f} init_seed={init_seed} sim_seed={sim_seed}",
            flush=True,
        )

        t0 = time.perf_counter()
        phi_run = sim.run(
            phi_init,
            param,
            NSTEPS,
            DT,
            NOISE,
            no_frames=NFRAMES,
            scheme=SCHEME,
            model=MODEL,
            verbatum=False,
            diagnostic_interval=SAVE_EVERY,
            reset_projection_diag=True,
            use_fastpath=USE_FASTPATH,
        )
        runtime = time.perf_counter() - t0

        tpost0 = time.perf_counter()
        arrays = extract_reduced_output(phi_run)
        proj_arrays, proj_summary = projection_arrays(sim, DT)
        arrays.update(proj_arrays)
        postprocess_runtime = time.perf_counter() - tpost0

        # Metadata sufficient to reconstruct the parameter point and important
        # numerical choices. Store as JSON text inside the same NPZ.
        metadata = {
            "task_id": int(task_id),
            "regime_index": int(regime_index),
            "regime": regime,
            "run_index": int(run_index),
            "dv_index": int(dv_index),
            "D_v": float(D_v),
            "D": D.tolist(),
            "Gamma": GAMMA.tolist(),
            "kappa": KAPPAS[regime].tolist(),
            "beta": float(BETA),
            "h": float(H),
            "noise_v": float(NOISE_V),
            "N": list(N),
            "L": list(L),
            "dt": float(DT),
            "nsteps": int(NSTEPS),
            "nframes_requested": int(NFRAMES),
            "nframes_stored": int(phi_run.shape[1]),
            "includes_initial_frame": True,
            "save_every_steps": int(SAVE_EVERY),
            "frame_dt": float(SAVE_EVERY * DT),
            "tail_frames": int(TAIL_FRAMES),
            "tail_std_ddof": int(TAIL_STD_DDOF),
            "rho_A0": float(RHO_A0),
            "rho_B0": float(RHO_B0),
            "initial_perturbation_std": float(INITIAL_PERTURBATION),
            "initialization": "independent Gaussian perturbations; unphysical cells redrawn until inside A>=0, B>=0, A+B<=1",
            "initial_bad_cells_first_pass": int(init_bad_cells),
            "initial_redraws_total": int(init_redraws),
            "init_seed": int(init_seed),
            "sim_seed": int(sim_seed),
            "noise": bool(NOISE),
            "scheme": SCHEME,
            "model": MODEL,
            "bc": BC,
            "fft": False,
            "schelling_flux": SCHELLING_FLUX,
            "projection_mode": str(getattr(sim, "projection_mode", PROJECTION_MODE)),
            "projection_floor": float(getattr(sim, "projection_floor", np.nan)),
            "projection_tol": float(getattr(sim, "projection_tol", np.nan)),
            "use_numba_projection": bool(USE_NUMBA_PROJECTION),
            "numba_projection_threads": int(NUMBA_PROJECTION_THREADS),
            "voter_noise_mode": VOTER_NOISE_MODE,
            "wf_gaussian_threshold": float(WF_GAUSSIAN_THRESHOLD),
            "reaction_diagnostics": bool(REACTION_DIAGNOSTICS),
            "use_fastpath": bool(USE_FASTPATH),
            "spectrum_transform": "DCT-II",
            "spectrum_norm": "ortho",
            "spectrum_centered_per_frame": bool(SPECTRUM_CENTERED),
            "spectrum_use_lattice_k": bool(SPECTRUM_USE_LATTICE_K),
            "spectrum_num_bins": int(N_SPECTRUM_BINS),
            "spectrum_pairs": list(SPECTRUM_NAMES),
            "k_mean_definition": "sum_k khat * |DCT(rho_A-rho_B)|^2 / sum_k |DCT(rho_A-rho_B)|^2",
            "interface_density_bc": BC,
            "runtime_seconds": float(runtime),
            "postprocess_runtime_seconds": float(postprocess_runtime),
            "projection_diagnostics": proj_summary,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "fhd_version": _fhd_version(),
            "git_commit": _git_commit(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
            "parameter_chunk_size": os.environ.get("PARAMETER_CHUNK_SIZE", ""),
        }

        arrays["metadata_json"] = _json_scalar(metadata)
        arrays["phi_initial"] = phi_init.copy()
        arrays["runtime_seconds"] = np.asarray(runtime)
        arrays["postprocess_runtime_seconds"] = np.asarray(postprocess_runtime)
        arrays["init_seed"] = np.asarray(init_seed, dtype=np.int64)
        arrays["initial_bad_cells_first_pass"] = np.asarray(init_bad_cells, dtype=np.int64)
        arrays["initial_redraws_total"] = np.asarray(init_redraws, dtype=np.int64)
        arrays["sim_seed"] = np.asarray(sim_seed, dtype=np.int64)
        arrays["D_v"] = np.asarray(D_v)
        arrays["D"] = D.copy()
        arrays["Gamma"] = GAMMA.copy()
        arrays["kappa"] = KAPPAS[regime].copy()
        arrays["beta"] = np.asarray(BETA)
        arrays["h"] = np.asarray(H)
        arrays["N"] = np.asarray(N, dtype=np.int64)
        arrays["L"] = np.asarray(L, dtype=float)
        arrays["dt"] = np.asarray(DT)
        arrays["nsteps"] = np.asarray(NSTEPS, dtype=np.int64)
        arrays["nframes_requested"] = np.asarray(NFRAMES, dtype=np.int64)
        arrays["nframes_stored"] = np.asarray(phi_run.shape[1], dtype=np.int64)
        arrays["save_every_steps"] = np.asarray(SAVE_EVERY, dtype=np.int64)

        atomic_savez_compressed(path, arrays)

        # Release the large trajectory before this worker takes another task.
        del phi_run, arrays
        gc.collect()

        print(
            f"[pid={pid}] SAVED task={task_id:03d} {path} "
            f"runtime={runtime:.1f}s post={postprocess_runtime:.1f}s",
            flush=True,
        )

    except Exception:
        print(f"ERROR in task {param_set}", flush=True)
        traceback.print_exc()
        raise


# =============================================================================
# Array-batched parameter sweep (same logic as previous cluster script)
# =============================================================================

parameter_sets = []
task_id = 0
for regime_index, regime in enumerate(REGIMES):
    for run_index in range(N_RUNS):
        for dv_index, D_v in enumerate(DV_VALS):
            parameter_sets.append(
                (task_id, regime_index, regime, run_index, dv_index, float(D_v))
            )
            task_id += 1

N_TOTAL = len(parameter_sets)

expected_total = len(REGIMES) * N_RUNS * len(DV_VALS)
assert N_TOTAL == expected_total

no_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
chunk_size = int(os.environ.get("PARAMETER_CHUNK_SIZE", 32))
array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

start = array_id * chunk_size
stop = min(start + chunk_size, N_TOTAL)
parameter_sets_this_job = parameter_sets[start:stop]

print(
    f"Array task {array_id}: parameter_sets[{start}:{stop}] "
    f"({len(parameter_sets_this_job)} simulations; total sweep={N_TOTAL})",
    flush=True,
)


def parallel_simulation(params):
    if len(params) == 0:
        print("No tasks assigned to this array job.", flush=True)
        return

    n_workers = min(no_cores, len(params))
    print(f"Number of simulations in this array job: {len(params)}", flush=True)
    print(f"Number of worker processes: {n_workers}", flush=True)

    # Keep workers alive across their assigned simulations so Numba compilation
    # can be reused. run_simulation explicitly deletes the large trajectory and
    # triggers garbage collection before a worker receives its next task.
    with multiprocessing.Pool(n_workers) as pool:
        pool.map(run_simulation, params)


if __name__ == "__main__":
    parallel_simulation(parameter_sets_this_job)
