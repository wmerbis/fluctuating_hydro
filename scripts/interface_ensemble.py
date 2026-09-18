#!/usr/bin/env python3
"""Local ensemble study of interface breakup in the Schelling+Voter FHD model.

Default: 32 stochastic Schelling+Voter realizations, at most 8 workers.
Completed runs are saved individually, so interrupted ensembles can be resumed.

Examples
--------
# Main experiment only
python interface_ensemble.py --runs 32 --workers 8

# Add a stochastic Schelling control and deterministic references
python interface_ensemble.py --runs 32 --workers 8 \
    --conditions sv_stoc schelling_stoc sv_det schelling_det

# Resume is automatic; use --overwrite to rerun completed files.
"""

# Prevent BLAS/OpenMP oversubscription before NumPy/SciPy/Numba are imported.
import os
for _var in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, label, generate_binary_structure

# This is the import used in the FHD repository in our earlier scripts.
# Change only this line if your local checkout exposes FHD_2D differently.
from fhd import FHD_2D as fhd


# -----------------------------------------------------------------------------
# Simulation parameters: copied from the requested setup
# -----------------------------------------------------------------------------
N = (128, 128)
L = (50.0, 50.0)
Lx, Ly = L
BC = "Neumann"

KAPPA = np.array([[0.6, -0.4],
                  [-0.4, 0.6]], dtype=float)
H = 0.01
D = 0.1 * np.ones(2)
GAMMA = np.eye(2)
D_V = 0.1
BETA = 10.0
PHI0 = 0.4

DT = 1.0e-3
NSTEPS = 1_000_000
FRAMES = 1000
BASE_SEED = 1234

# Coarse-graining scales, in grid cells, at which the interface perimeter is
# measured.  A smooth interface has nearly scale-independent perimeter; a rough
# or fractal-like boundary loses perimeter as it is smoothed.
INTERFACE_SIGMAS = (0.0, 1.0, 2.0, 4.0)
DOMAIN_SIGMA = 1.0
SNAPSHOT_FRACTIONS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)

# Default experiment plus useful controls.
#   sv_stoc        : requested setup
#   schelling_stoc : same stochastic solver but with the voter channel removed
#   sv_det         : same voter current, with stochastic terms disabled
#   schelling_det  : pure deterministic Schelling reference interface
CONDITIONS = {
    "sv_stoc":        {"D_v": D_V, "noise": True},
    "schelling_stoc": {"D_v": 0.0, "noise": True},
    "sv_det":         {"D_v": D_V, "noise": False},
    "schelling_det":  {"D_v": 0.0, "noise": False},
}


# -----------------------------------------------------------------------------
# Model construction
# -----------------------------------------------------------------------------
def build_simulation(seed: int, d_v: float):
    sim_2d = fhd.fhd_2d(
        L, N,
        bc=BC,
        fft=False,
        schelling_flux="finite_volume",
        projection_mode="redistribute",
        use_numba_projection=True,
        numba_projection_threads=1,
        voter_noise_mode="wright_fisher",
        wf_gaussian_threshold=0.01,
    )
    sim_2d.set_seed(seed)

    param = {
        "D": D.copy(),
        "kappa": KAPPA.copy(),
        "Gamma": GAMMA.copy(),
        "D_v": float(d_v),
        "beta": BETA,
        "h": H,
    }

    x = sim_2d.x
    y = sim_2d.y
    _, yy = np.meshgrid(x, y, indexing="ij")

    phi = np.zeros((sim_2d.nspecies,) + N, dtype=float)
    phi[0] = PHI0 * (1.0 + np.tanh(yy))
    phi[1] = PHI0 * (1.0 - np.tanh(yy))
    return sim_2d, param, phi


# -----------------------------------------------------------------------------
# Observables
# -----------------------------------------------------------------------------
def voter_interface_density_frame(phi: np.ndarray, eps: float = 1.0e-14) -> float:
    """Correlation-based Voter interface density, ``1 - <m_i m_j>``.

    Here ``m = (rho_A-rho_B)/(rho_A+rho_B)`` and the average is over
    nearest-neighbour bonds.  For Neumann boundaries only physical interior
    bonds are counted; periodic runs wrap with ``np.roll``.

    This is the interface observable used in the earlier cluster runs.  Note
    that, for diffuse fields, it is not identical to a geometric perimeter:
    cells with |m| < 1 also lower the bond correlation even without a sharp
    sign change.
    """
    rho_a, rho_b = phi
    occ = rho_a + rho_b
    m = np.divide(
        rho_a - rho_b, occ,
        out=np.zeros_like(rho_a),
        where=occ > eps,
    )

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


def _interface_length_density(composition, dx, dy, sigma_cells=0.0):
    """Grid estimator of A/B interface length divided by physical area.

    composition = (rho_A-rho_B)/(rho_A+rho_B).  We optionally smooth it before
    thresholding, then count nearest-neighbour sign changes.  Neumann boundaries
    are respected by *not* wrapping across the box edges.
    """
    if sigma_cells > 0:
        field = gaussian_filter(composition, sigma=sigma_cells, mode="reflect")
    else:
        field = composition

    phase_a = field >= 0.0

    # axis 0 is x: crossing between x-neighbours contributes a segment of dy.
    n_x = np.count_nonzero(phase_a[1:, :] != phase_a[:-1, :])
    # axis 1 is y: crossing between y-neighbours contributes a segment of dx.
    n_y = np.count_nonzero(phase_a[:, 1:] != phase_a[:, :-1])

    perimeter = n_x * dy + n_y * dx
    return perimeter / (Lx * Ly)


def _domain_statistics(composition, sigma_cells=1.0):
    """Number of connected A/B domains and largest-domain area fraction."""
    field = gaussian_filter(composition, sigma=sigma_cells, mode="reflect")
    phase_a = field >= 0.0
    conn4 = generate_binary_structure(2, 1)

    lab_a, n_a = label(phase_a, structure=conn4)
    lab_b, n_b = label(~phase_a, structure=conn4)

    largest = 0
    if n_a:
        largest = max(largest, int(np.bincount(lab_a.ravel())[1:].max()))
    if n_b:
        largest = max(largest, int(np.bincount(lab_b.ravel())[1:].max()))

    return float(n_a + n_b), largest / phase_a.size


def _relative_entropy_with_vacancies(rho_a, rho_b, eps=1e-12):
    """Local-vs-global normalized KL measure used as a pattern diagnostic."""
    rho_0 = 1.0 - rho_a - rho_b
    local = np.stack((rho_a, rho_b, rho_0), axis=0)
    local = np.clip(local, eps, None)
    global_dist = np.clip(local.mean(axis=(1, 2)), eps, None)

    h_global = -np.sum(global_dist * np.log(global_dist))
    if h_global <= eps:
        return np.nan

    kl = np.sum(
        local * np.log(local / global_dist[:, None, None]), axis=0
    )
    return float(np.mean(kl) / h_global)


def _spectral_scale(composition, kgrid):
    """Return k_mean and ell=2*pi/k_mean for the centered composition field."""
    f = composition - composition.mean()
    power = np.abs(np.fft.fft2(f)) ** 2
    power[0, 0] = 0.0
    z = power.sum()
    if z <= 0.0:
        return np.nan, np.nan
    k_mean = float(np.sum(kgrid * power) / z)
    ell = float(2.0 * np.pi / k_mean) if k_mean > 0 else np.nan
    return k_mean, ell


def compute_observables(phi_run):
    """Reduce an entire trajectory to inexpensive time series + a few snapshots."""
    nt = phi_run.shape[1]
    dx = Lx / N[0]
    dy = Ly / N[1]
    eps = 1e-12

    kx = 2.0 * np.pi * np.fft.fftfreq(N[0], d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(N[1], d=dy)
    kxx, kyy = np.meshgrid(kx, ky, indexing="ij")
    kgrid = np.sqrt(kxx**2 + kyy**2)

    names = [
        "rhoA_mean", "rhoB_mean", "occupancy_mean", "magnetization",
        "overlap", "relative_entropy", "total_variation",
        "voter_interface_density",
        "n_domains", "largest_domain_fraction",
        "k_mean", "domain_scale",
        "effective_boundary_dimension", "roughness_ratio",
    ]
    obs = {name: np.empty(nt, dtype=float) for name in names}
    for s in INTERFACE_SIGMAS:
        obs[f"interface_density_sigma{s:g}"] = np.empty(nt, dtype=float)

    for t in range(nt):
        rho_a = phi_run[0, t]
        rho_b = phi_run[1, t]
        occupancy = rho_a + rho_b
        composition = (rho_a - rho_b) / np.maximum(occupancy, eps)

        obs["rhoA_mean"][t] = rho_a.mean()
        obs["rhoB_mean"][t] = rho_b.mean()
        obs["occupancy_mean"][t] = occupancy.mean()
        obs["magnetization"][t] = np.sum(rho_a - rho_b) / max(np.sum(occupancy), eps)

        # 0 in locally pure A/B regions, 1 in a 50-50 local mixture.  Occupancy
        # weighting avoids letting nearly-empty cells dominate the statistic.
        local_overlap = 4.0 * rho_a * rho_b / np.maximum(occupancy**2, eps)
        obs["overlap"][t] = np.sum(occupancy * local_overlap) / max(np.sum(occupancy), eps)

        obs["relative_entropy"][t] = _relative_entropy_with_vacancies(rho_a, rho_b)

        # Legacy nearest-neighbour correlation measure used in the cluster runs.
        # Keep this alongside the geometric interface-length estimator below so
        # their behavior can be compared directly.
        obs["voter_interface_density"][t] = voter_interface_density_frame(
            phi_run[:2, t]
        )

        # Smooth at one cell before the continuum total-variation diagnostic.
        c1 = gaussian_filter(composition, sigma=1.0, mode="reflect")
        dc_dx, dc_dy = np.gradient(c1, dx, dy, edge_order=2)
        obs["total_variation"][t] = np.mean(np.hypot(dc_dx, dc_dy))

        perimeters = []
        for s in INTERFACE_SIGMAS:
            val = _interface_length_density(composition, dx, dy, s)
            obs[f"interface_density_sigma{s:g}"][t] = val
            perimeters.append(val)

        obs["n_domains"][t], obs["largest_domain_fraction"][t] = \
            _domain_statistics(composition, DOMAIN_SIGMA)

        obs["k_mean"][t], obs["domain_scale"][t] = _spectral_scale(composition, kgrid)

        # Effective boundary dimension from P(epsilon) ~ epsilon^(1-D_eff).
        # sigma=0 is excluded from the fit because log(0) is undefined.
        scales = np.array(INTERFACE_SIGMAS[1:], dtype=float) * np.sqrt(dx * dy)
        pfit = np.array(perimeters[1:], dtype=float)
        good = pfit > 0
        if np.count_nonzero(good) >= 2:
            slope = np.polyfit(np.log(scales[good]), np.log(pfit[good]), 1)[0]
            obs["effective_boundary_dimension"][t] = 1.0 - slope
        else:
            obs["effective_boundary_dimension"][t] = np.nan

        obs["roughness_ratio"][t] = (
            perimeters[0] / perimeters[-1] if perimeters[-1] > 0 else np.nan
        )

    # A normalized breakup measure is convenient for plots and comparisons.
    p1 = obs["interface_density_sigma1"]
    obs["interface_ratio_sigma1"] = p1 / max(p1[0], eps)

    voter_if = obs["voter_interface_density"]
    obs["voter_interface_ratio"] = voter_if / max(voter_if[0], eps)

    # The run() routine returns evenly spaced frames in the existing workflow.
    times = np.linspace(0.0, NSTEPS * DT, nt)

    snap_idx = np.unique(
        np.rint(np.array(SNAPSHOT_FRACTIONS) * (nt - 1)).astype(int)
    )
    snapshots = phi_run[:, snap_idx].astype(np.float32, copy=True)
    snapshot_times = times[snap_idx]

    return times, obs, snap_idx, snapshot_times, snapshots


# -----------------------------------------------------------------------------
# One realization
# -----------------------------------------------------------------------------
def run_one(condition, run_id, seed, out_file):
    settings = CONDITIONS[condition]
    sim_2d, param, phi = build_simulation(seed, settings["D_v"])

    phi_run = sim_2d.run(
        phi,
        param,
        NSTEPS,
        DT,
        settings["noise"],
        FRAMES,
        scheme="FE",
        model="Schelling+Voter",
        use_fastpath=True,
        verbatum=False,
    )
    phi_run = np.asarray(phi_run)

    if phi_run.ndim != 4 or phi_run.shape[0] < 2:
        raise RuntimeError(
            f"Unexpected phi_run shape {phi_run.shape}; expected "
            "(species, frames, Nx, Ny)."
        )

    times, obs, snap_idx, snapshot_times, snapshots = compute_observables(phi_run)

    payload = {
        "condition": np.array(condition),
        "run_id": np.array(run_id),
        "seed": np.array(seed),
        "time": times,
        "snapshot_indices": snap_idx,
        "snapshot_times": snapshot_times,
        "snapshots": snapshots,
    }
    payload.update(obs)

    out_file = Path(out_file)
    tmp = out_file.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, out_file)
    return str(out_file)


# -----------------------------------------------------------------------------
# Aggregation and plotting
# -----------------------------------------------------------------------------
NON_METRIC_KEYS = {
    "condition", "run_id", "seed", "time",
    "snapshot_indices", "snapshot_times", "snapshots",
}


def aggregate_condition(condition_dir):
    files = sorted(condition_dir.glob("run_*.npz"))
    if not files:
        return None

    with np.load(files[0], allow_pickle=False) as z0:
        times = z0["time"].copy()
        metric_names = [k for k in z0.files if k not in NON_METRIC_KEYS]

    stacks = {name: [] for name in metric_names}
    summaries = []

    for path in files:
        with np.load(path, allow_pickle=False) as z:
            if not np.allclose(z["time"], times):
                raise RuntimeError(f"Frame times differ in {path}")
            row = {
                "run_id": int(z["run_id"]),
                "seed": int(z["seed"]),
            }
            for name in metric_names:
                a = np.asarray(z[name], dtype=float)
                stacks[name].append(a)

                if name in {
                    "interface_density_sigma1", "interface_ratio_sigma1",
                    "voter_interface_density", "voter_interface_ratio",
                    "effective_boundary_dimension", "roughness_ratio",
                    "n_domains", "overlap", "domain_scale", "magnetization",
                    "occupancy_mean",
                }:
                    tail = a[max(0, int(0.8 * len(a))):]
                    row[f"{name}_final"] = float(a[-1])
                    row[f"{name}_late_mean"] = float(np.nanmean(tail))
            summaries.append(row)

    agg = {"time": times, "n_runs": np.array(len(files))}
    for name, arrays in stacks.items():
        x = np.stack(arrays, axis=0)
        agg[f"{name}_mean"] = np.nanmean(x, axis=0)
        agg[f"{name}_std"] = np.nanstd(x, axis=0, ddof=1) if len(files) > 1 else np.zeros_like(times)
        agg[f"{name}_sem"] = agg[f"{name}_std"] / np.sqrt(len(files))
        agg[f"{name}_q10"] = np.nanquantile(x, 0.10, axis=0)
        agg[f"{name}_q90"] = np.nanquantile(x, 0.90, axis=0)

    np.savez_compressed(condition_dir / "ensemble_metrics.npz", **agg)

    # Wide time-series CSV: mean/std/SEM for every observable.
    csv_path = condition_dir / "ensemble_timeseries.csv"
    keys = ["time"] + [k for k in agg.keys() if k not in {"time", "n_runs"}]
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i, t in enumerate(times):
            writer.writerow([t] + [float(np.asarray(agg[k])[i]) for k in keys[1:]])

    # Compact per-realization late-time/final summary.
    if summaries:
        summary_keys = list(summaries[0].keys())
        with (condition_dir / "run_summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_keys)
            writer.writeheader()
            writer.writerows(summaries)

    _plot_condition(condition_dir, agg)
    _plot_interface_estimators(condition_dir, agg)
    _plot_snapshots(files[0], condition_dir / "snapshots_run000.png")
    return agg


def _plot_condition(condition_dir, agg):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    t = agg["time"]
    panels = [
        ("interface_ratio_sigma1", "interface length / initial"),
        ("effective_boundary_dimension", r"effective boundary $D_{eff}$"),
        ("n_domains", "number of A/B domains"),
        ("overlap", "A/B overlap"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (name, ylabel) in zip(axes.ravel(), panels):
        mean = agg[f"{name}_mean"]
        sem = agg[f"{name}_sem"]
        ax.plot(t, mean)
        ax.fill_between(t, mean - sem, mean + sem, alpha=0.25)
        ax.set_xlabel("time")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    fig.suptitle(condition_dir.name)
    fig.savefig(condition_dir / "summary_metrics.png", dpi=180)
    plt.close(fig)


def _plot_interface_estimators(condition_dir, agg):
    """Compare geometric perimeter and legacy bond-correlation observables."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    t = agg["time"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # Raw observables have different units/normalizations, so show them together
    # only for qualitative time dependence.
    axes[0].plot(t, agg["interface_density_sigma1_mean"], label="geometric, sigma=1")
    axes[0].plot(t, agg["voter_interface_density_mean"], label=r"legacy $1-\langle m_i m_j\rangle$")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("interface observable")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    # Normalizing each measure by its own initial value makes their relative
    # response to interface breakup directly comparable.
    for name, label in (
        ("interface_ratio_sigma1", "geometric / initial"),
        ("voter_interface_ratio", "legacy / initial"),
    ):
        mean = agg[f"{name}_mean"]
        sem = agg[f"{name}_sem"]
        axes[1].plot(t, mean, label=label)
        axes[1].fill_between(t, mean - sem, mean + sem, alpha=0.18)
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("observable / initial")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    fig.suptitle(condition_dir.name)
    fig.savefig(condition_dir / "interface_estimator_comparison.png", dpi=180)
    plt.close(fig)


def _plot_snapshots(run_file, out_png):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    with np.load(run_file, allow_pickle=False) as z:
        snaps = z["snapshots"]
        times = z["snapshot_times"]

    ns = snaps.shape[1]
    fig, axes = plt.subplots(2, int(np.ceil(ns / 2)), figsize=(12, 6), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for i in range(ns):
        a, b = snaps[0, i], snaps[1, i]
        c = (a - b) / np.maximum(a + b, 1e-12)
        im = axes[i].imshow(
            c.T, origin="lower", extent=(0, Lx, 0, Ly),
            vmin=-1, vmax=1, cmap="coolwarm", aspect="equal",
        )
        axes[i].set_title(f"t={times[i]:.1f}")
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("y")
    for ax in axes[ns:]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label=r"$(\rho_A-\rho_B)/(\rho_A+\rho_B)$")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_condition_comparison(output_root, conditions, aggregates):
    if len(conditions) < 2:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for condition in conditions:
        agg = aggregates.get(condition)
        if agg is None:
            continue
        t = agg["time"]
        mean = agg["interface_ratio_sigma1_mean"]
        sem = agg["interface_ratio_sigma1_sem"]
        ax.plot(t, mean, label=condition)
        ax.fill_between(t, mean - sem, mean + sem, alpha=0.18)
    ax.set_xlabel("time")
    ax.set_ylabel("interface length / initial")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(output_root / "comparison_interface_breakup.png", dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Resume compatibility
# -----------------------------------------------------------------------------
def _run_file_has_current_observables(path: Path) -> bool:
    """Return True if a saved run contains all observables required now.

    This makes resume behavior safe after extending the analysis: run files made
    with an older version of this script are automatically recomputed rather
    than silently omitted from the new interface comparison.
    """
    try:
        with np.load(path, allow_pickle=False) as z:
            required = {
                "voter_interface_density",
                "voter_interface_ratio",
                "interface_density_sigma1",
                "interface_ratio_sigma1",
            }
            return required.issubset(z.files)
    except Exception:
        return False


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=32, help="realizations per condition")
    p.add_argument("--workers", type=int, default=8, help="parallel workers, capped at 8")
    p.add_argument(
        "--conditions", nargs="+", default=["sv_stoc"],
        choices=tuple(CONDITIONS),
        help="simulation conditions to run",
    )
    p.add_argument("--output", type=Path, default=Path("interface_ensemble"))
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    max_workers = min(
        max(1, args.workers),
        8,
        max(1, args.runs * len(args.conditions)),
        os.cpu_count() or 1,
    )

    metadata = {
        "N": N, "L": L, "bc": BC, "kappa": KAPPA.tolist(), "h": H,
        "D": D.tolist(), "Gamma": GAMMA.tolist(), "D_v": D_V,
        "beta": BETA, "phi0": PHI0, "dt": DT, "nsteps": NSTEPS,
        "frames": FRAMES, "base_seed": BASE_SEED,
        "interface_sigmas_cells": INTERFACE_SIGMAS,
        "conditions": CONDITIONS,
        "stochastic_runs_per_condition": args.runs,
        "deterministic_runs_per_condition": 1,
        "workers": max_workers,
    }
    with (args.output / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    jobs = []
    for condition in args.conditions:
        cdir = args.output / condition
        cdir.mkdir(parents=True, exist_ok=True)
        n_condition_runs = args.runs if CONDITIONS[condition]["noise"] else 1
        for run_id in range(n_condition_runs):
            out_file = cdir / f"run_{run_id:03d}.npz"
            if out_file.exists() and not args.overwrite:
                if _run_file_has_current_observables(out_file):
                    continue
                print(f"[{condition}] {out_file.name} uses an older observable set; recomputing.")
            # Same run_id gets the same seed in each condition; run 0 is 1234.
            seed = BASE_SEED + run_id
            jobs.append((condition, run_id, seed, out_file))

    print(f"Using {max_workers} worker(s); {len(jobs)} run(s) still need to be computed.")

    if jobs:
        # spawn is robust with Numba/OpenMP and also works on macOS/Windows.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
            future_map = {
                pool.submit(run_one, condition, run_id, seed, str(out_file)):
                    (condition, run_id)
                for condition, run_id, seed, out_file in jobs
            }
            for fut in as_completed(future_map):
                condition, run_id = future_map[fut]
                try:
                    path = fut.result()
                    print(f"[{condition}] run {run_id:03d} complete -> {path}")
                except Exception as exc:
                    print(f"[{condition}] run {run_id:03d} FAILED: {exc}")
                    raise

    aggregates = {}
    for condition in args.conditions:
        aggregates[condition] = aggregate_condition(args.output / condition)

    plot_condition_comparison(args.output, args.conditions, aggregates)
    print(f"Done. Results are in: {args.output.resolve()}")


if __name__ == "__main__":
    main()
