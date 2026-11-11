#!/usr/bin/env python3
"""Pure Schelling phase sweep: separate recordings, observables and grids.

Stage 1 (SIMULATE): each simulation writes exactly one self-contained NPZ:
    OUT/recordings/{reciprocal,nonreciprocal}/b{ib}_x{ix}_r{rep}.npz
    Contains low-k Neumann DCT history + sparse full-field snapshots + metadata.

Stage 2 (ANALYZE): reads the recordings; writes derived observables to:
    OUT/{reciprocal,nonreciprocal}/b{ib}_x{ix}_r{rep}.npz
    Includes the whole equal-time S_ab(k,t), the C_ab(k,tau), S_ab(k,omega),
    eigenvalue indicators and scalar order parameters.  No simulation needed.

Stage 3 (AGGREGATE): averages scalar per-run observables into:
    OUT/reciprocal_grids.npz and OUT/nonreciprocal_grids.npz
    Each family retains its own x axis. This is compatible with
    scripts/plot_schelling_phase_diagram.py.

Examples, run from the repository root:

  # Local 4-core smoke test, 2 parameter points, 50,000 steps each:
  python scripts/schelling_phase_sweep.py simulate --out phase_sweep_test \\
      --families reciprocal nonreciprocal --beta-values 8 \\
      --x-values 0.6 --repeats 1 --workers 4 --dt 0.001 \\
      --nsteps 50000 --mode-every 100 --snapshots 500
  python scripts/schelling_phase_sweep.py analyze --out phase_sweep_test
  python scripts/schelling_phase_sweep.py aggregate --out phase_sweep_test

  # Both families in one sweep: shared beta axis, family-specific x axes:
  # reciprocal [-1,1], nonreciprocal [0,1] by default.
  python scripts/schelling_phase_sweep.py simulate --out phase_sweep_21 \
      --families reciprocal nonreciprocal --beta-min 0 --beta-max 8 \
      --n-beta 21 --n-x 21 --workers 4

  # Or run all three stages consecutively with the same arguments:
  python scripts/schelling_phase_sweep.py all --out phase_sweep_test \\
      --families nonreciprocal --beta-values 8 --x-values 0.6 --workers 4

Important:
  * Assumes the modified fhd.fhd_2d.run accepts mode_recorder= and
    store_full_run=False, and that fhd/low_k_recording.py is installed.
  * Neumann boundaries are required for real DCT mode recording.
  * min_cycles=0: preserve raw complex eigenvalues; report N_cycles separately.
  * A beta=0 analytic reference uses cell_capacity=(dx*dy)/h**2.
  * Dynamic statistics are evaluated on saved mode amplitudes; spatial
    statistics and late-time coarsening use full-grid sparse snapshots.
"""
from __future__ import annotations

import os
# Avoid accidentally requesting many BLAS/FFT threads in every worker.
for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_env] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import re
import sys
import time
import warnings

import numpy as np

# Allow invocation as a standalone script from any working directory.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAMILIES = ("reciprocal", "nonreciprocal")
REC_NAME = re.compile(r"b(\d+)_x(\d+)_r(\d+)\.npz$")
EIGEN_KWARGS = dict(
    k_range=(0.2, 2.5),
    widths=(0.25, 0.35, 0.5),
    step=0.05,
    bin_width=0.1,
    lag_frames=1,
    min_modes=20,
    max_condition=1e6,
    min_modulus=0.25,
    min_cycles=0.0,
)
GRID_SCALARS = (
    "C_late", "P_plus_late", "P_minus_late", "I_seg_late",
    "I_seg_excess_late", "I_occ_late", "entropy_kl_late",
    "normalized_entropy_segregation_late", "temporal_antisymmetry",
    "oscillation_negative_lobe", "r_max", "Omega_max",
    "r_at_Omega_max", "k_r_max", "k_Omega_max", "r_complex_max",
    "Omega_at_r_complex_max", "k_r_complex_max", "N_cycles",
    "N_cycles_at_Omega_max", "r_complex_early", "Omega_complex_early",
    "r_complex_late", "Omega_complex_late", "N_cycles_early",
    "N_cycles_late", "k_peak_minus_late", "k_peak_plus_late",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("simulate", "analyze", "aggregate", "all"))
    p.add_argument("--out", type=Path, default=Path("phase_sweep"))
    p.add_argument("--families", nargs="+", choices=FAMILIES,
                   default=list(FAMILIES))
    p.add_argument("--beta-values", type=float, nargs="+", default=None,
                   help="Explicit beta grid, e.g. 0 2 4 6 8")
    p.add_argument("--beta-min", type=float, default=0.)
    p.add_argument("--beta-max", type=float, default=8.)
    p.add_argument("--n-beta", type=int, default=9)
    p.add_argument("--x-values", type=float, nargs="+", default=None,
                   help="Explicit x axis for BOTH families; overrides family defaults")
    p.add_argument("--x-min", type=float, default=None,
                   help="Override lower x bound for BOTH families (defaults: reciprocal=-1, nonreciprocal=0)")
    p.add_argument("--x-max", type=float, default=None,
                   help="Override upper x bound for BOTH families (default: 1)")
    p.add_argument("--n-x", type=int, default=11)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--workers", type=int, default=4,
                   help="Number of independent worker processes (local default 4)")
    p.add_argument("--N", type=int, nargs=2, default=(128, 128))
    p.add_argument("--L", type=float, nargs=2, default=(50., 50.))
    p.add_argument("--rho", type=float, nargs=2, default=(.35, .35))
    p.add_argument("--D", type=float, nargs=2, default=(1., 1.))
    p.add_argument("--sigma", type=float, default=.5)
    p.add_argument("--people-per-cell", type=float, default=400.)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--nsteps", type=int, default=50_000)
    p.add_argument("--mode-every", type=int, default=100)
    p.add_argument("--snapshots", type=int, default=500,
                   help="Saved full-field INTERVALS; produces snapshots+1 fields")
    p.add_argument("--mode-kmax", type=float, default=3.5)
    p.add_argument("--initial-std", type=float, default=.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true",
                   help="Rerun/replace existing recordings or derived results")
    p.add_argument("--limit", type=int, default=None,
                   help="Max tasks in this invocation, useful for smoke tests")
    # Analysis options deliberately independent of the simulation cadence.
    p.add_argument("--burn-time", type=float, default=10.,
                   help="Physical burn-in for temporal observables")
    p.add_argument("--late-fraction", type=float, default=.2,
                   help="Fraction of spatial snapshots used for late-time summaries")
    p.add_argument("--spatial-bins", type=int, default=160)
    p.add_argument("--dynamic-bins", type=int, default=60)
    p.add_argument("--excess-k-band", type=float, nargs=2,
                   default=(.2, 2.5))
    p.add_argument("--eig-k-range", type=float, nargs=2, default=(.2, 2.5))
    p.add_argument("--eig-min-modes", type=int, default=20)
    p.add_argument("--eig-min-modulus", type=float, default=.25)
    p.add_argument("--no-full-dynamics", action="store_true",
                   help="Do not save full C(k,tau), S(k,omega), only scalars")
    return p.parse_args(argv)


def _check_args(args):
    if args.workers < 1 or args.repeats < 1:
        raise ValueError("workers and repeats must be positive")
    if args.nsteps < 1 or args.dt <= 0 or args.mode_every < 1:
        raise ValueError("Need positive nsteps, dt, mode_every")
    if args.snapshots < 1 or args.nsteps % args.snapshots:
        raise ValueError("nsteps must be a multiple of snapshots to store exactly snapshots+1 equally spaced fields")
    if any(n < 4 for n in args.N) or any(x <= 0 for x in args.L):
        raise ValueError("N>=4 and positive L required")
    if any(x <= 0 for x in args.D) or args.people_per_cell <= 1:
        raise ValueError("D must be positive and people-per-cell > 1")
    if min(args.rho) < 0 or sum(args.rho) >= 1:
        raise ValueError("rho must lie in the interior of density simplex")
    if args.mode_kmax <= max(0., args.eig_k_range[1]):
        raise ValueError("mode-kmax must exceed the upper eigen k range")
    if not 0 <= args.burn_time < args.nsteps * args.dt:
        raise ValueError("burn-time must lie inside the simulation")
    if not 0 < args.late_fraction <= 1 or args.initial_std < 0:
        raise ValueError("late-fraction/initial-std invalid")
    if not 0 < args.eig_k_range[0] < args.eig_k_range[1]:
        raise ValueError("eig-k-range must be strictly positive and ascending")
    if not 0 <= args.excess_k_band[0] < args.excess_k_band[1]:
        raise ValueError("excess-k-band must satisfy 0<=lower<upper")
    if args.spatial_bins < 1 or args.dynamic_bins < 1:
        raise ValueError("Need >=1 spectral bins")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")


def _grid_config(args):
    betas = (np.asarray(args.beta_values, float) if args.beta_values is not None
             else np.linspace(args.beta_min, args.beta_max, args.n_beta))
    if not len(betas) or not np.isfinite(betas).all() or np.any(np.diff(betas) <= 0):
        raise ValueError("beta grid must be nonempty, finite, and strictly increasing")
    if args.x_values is None and args.n_x < 1:
        raise ValueError("--n-x must be >= 1")
    # Each family has its own x axis by default, just like the original
    # Schelling sweep. Explicit --x-values/--x-min/--x-max override these
    # defaults for BOTH families, retaining the old CLI's override semantics.
    x_by_family = {}
    for family in args.families:
        if args.x_values is not None:
            xs = np.asarray(args.x_values, dtype=float)
        else:
            x_min = (args.x_min if args.x_min is not None
                     else (-1. if family == "reciprocal" else 0.))
            x_max = args.x_max if args.x_max is not None else 1.
            xs = np.linspace(x_min, x_max, args.n_x)
        if not len(xs) or not np.isfinite(xs).all() or np.any(np.diff(xs) <= 0):
            raise ValueError(f"{family} x grid must be nonempty, finite, and strictly increasing")
        x_by_family[family] = xs.tolist()
    L = list(map(float, args.L)); N = list(map(int, args.N))
    # For an anisotropic grid the microscopic area is the natural choice.
    h = np.sqrt((L[0] / N[0]) * (L[1] / N[1]) / args.people_per_cell)
    return dict(
        version=2, families=list(args.families), beta=betas.tolist(),
        x_by_family=x_by_family, N=N, L=L, bc="Neumann",
        rho=list(map(float, args.rho)), D=list(map(float, args.D)),
        sigma=float(args.sigma), rho_a=float(args.rho[0]), rho_b=float(args.rho[1]),
        people_per_cell=float(args.people_per_cell),
        h=float(h), dt=float(args.dt), nsteps=int(args.nsteps),
        mode_every=int(args.mode_every),
        snapshot_every=int(args.nsteps // args.snapshots),
        snapshots=int(args.snapshots), mode_k_band=[0., float(args.mode_kmax)],
        initial_std=float(args.initial_std), seed=int(args.seed),
    )


def _xs_for_family(cfg, family):
    """Per-family x grid; also read recordings from pre-v2 sweep configs."""
    if "x_by_family" in cfg:
        return np.asarray(cfg["x_by_family"][family], dtype=float)
    return np.asarray(cfg["x"], dtype=float)


def _load_config(out):
    path = out / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}; run the simulate stage first")
    return json.loads(path.read_text())


def _write_config(out, new):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "config.json"
    if path.exists():
        current = json.loads(path.read_text())
        if current != new:
            raise ValueError(f"{path} has a different simulation configuration. Use a NEW --out directory to avoid mixing runs")
    else:
        path.write_text(json.dumps(new, indent=2) + "\n")


def _kappa(family, x):
    return np.array([[1., x], [x if family == "reciprocal" else -x, 1.]], dtype=float)


def _atomic_npz(path, payload, compressed=True):
    """Write to a temporary file and rename only when complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temp.open("wb") as fh:
            if compressed:
                np.savez_compressed(fh, **payload)
            else:
                np.savez(fh, **payload)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _run_path(out, family, ib, ix, rep, *, derived=False):
    base = out / ("" if derived else "recordings") / family
    return base / f"b{ib}_x{ix}_r{rep}.npz"


def _seed_for_job(seed, family, ib, ix, rep):
    # Stable across process counts and schedules.
    from numpy.random import SeedSequence
    seq = SeedSequence([int(seed), FAMILIES.index(family), int(ib), int(ix), int(rep)])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def _simulate_one(job):
    cfg, out_str, family, ib, ix, rep, overwrite = job
    out = Path(out_str)
    path = _run_path(out, family, ib, ix, rep)
    if path.exists() and not overwrite:
        return f"SKIP {path}"
    # Import the simulator in the worker after environment/thread caps are set.
    import fhd
    from fhd.low_k_recording import LowKRecorder

    N = tuple(cfg["N"]); L = tuple(cfg["L"])
    beta = float(cfg["beta"][ib]); x = float(_xs_for_family(cfg, family)[ix])
    kappa = _kappa(family, x)
    seed = _seed_for_job(cfg["seed"], family, ib, ix, rep)
    rng = np.random.default_rng(seed)
    rho = np.asarray(cfg["rho"], dtype=float)
    phi = rho[:, None, None] + cfg["initial_std"] * rng.normal(size=(2, *N))

    sim = fhd.fhd_2d(
        L, N, bc="Neumann", fft=False,
        schelling_flux="finite_volume", projection_mode="redistribute",
        use_numba_projection=True, numba_projection_threads=1,
        voter_noise_mode="wright_fisher", wf_gaussian_threshold=.01,
        reaction_diagnostic=False,
    )
    sim.set_seed(seed)
    phi = sim.project_density_fast(phi)
    recorder = LowKRecorder(
        N=N, L=L, nsteps=cfg["nsteps"], dt=cfg["dt"],
        k_band=tuple(cfg["mode_k_band"]),
        mode_every=cfg["mode_every"],
        snapshot_every=cfg["snapshot_every"], dtype=np.float32,
    )
    param = dict(
        D=np.asarray(cfg["D"], dtype=float),
        kappa=kappa, beta=beta, h=cfg["h"], sigma=cfg["sigma"],
        rho_bar=rho.copy(), theta_face="midpoint",
        schelling_noise_mobility="symmetric",
    )
    start = time.perf_counter()
    # The requested user-integrated optional solver hooks.
    final = sim.run(
        phi, param, cfg["nsteps"], cfg["dt"], True,
        cfg["snapshots"], scheme="FE", model="Schelling",
        use_fastpath=False, verbatum=False,
        mode_recorder=recorder, store_full_run=False,
    )
    elapsed = time.perf_counter() - start
    result = recorder.result()
    if len(result["snapshot_steps"]) != cfg["snapshots"] + 1:
        raise RuntimeError("Snapshot cadence unexpectedly lost frames")
    if not np.array_equal(result["snapshot_steps"],
                          np.arange(cfg["snapshots"] + 1) * cfg["snapshot_every"]):
        raise RuntimeError("Snapshot time indices do not match requested cadence")
    if result["mode_steps"][0] != 0 or not np.all(np.diff(result["mode_steps"]) == cfg["mode_every"]):
        raise RuntimeError("Mode time indices are not uniformly spaced")
    result.update(
        record_version=np.int64(1),
        family=np.array(family), beta=np.float64(beta), x=np.float64(x),
        ib=np.int32(ib), ix=np.int32(ix), rep=np.int32(rep),
        seed=np.uint32(seed), D=np.asarray(cfg["D"], dtype=float),
        kappa=kappa, rho=np.asarray(cfg["rho"], float),
        h=np.float64(cfg["h"]), sigma=np.float64(cfg["sigma"]),
        people_per_cell=np.float64(cfg["people_per_cell"]),
        nsteps=np.int64(cfg["nsteps"]), dt=np.float64(cfg["dt"]),
        mode_every=np.int32(cfg["mode_every"]), noise=np.bool_(True),
        snapshot_every=np.int32(cfg["snapshot_every"]),
        mode_times=result["mode_steps"] * cfg["dt"],
        snapshot_times=result["snapshot_steps"] * cfg["dt"],
        final_mean=np.asarray(final).mean(axis=(-2, -1)),
        elapsed_sec=np.float64(elapsed),
    )
    _atomic_npz(path, result)
    return f"OK {family} beta={beta:.3g} x={x:.3g} rep={rep} {elapsed:.1f}s -> {path}"


def _entropy_kl_per_frame(snapshots):
    """Mean local KL to global A/B/vacancy fractions, one scalar per frame."""
    nt = snapshots.shape[1]
    kl = np.empty(nt, float)
    norm = np.empty(nt, float)
    # Stream over individual snapshots; no enormous (T,Nx,Ny,3) tensor.
    for t in range(nt):
        a = np.asarray(snapshots[:, t], dtype=float)
        vac = 1. - a.sum(axis=0)
        p = np.stack((a[0], a[1], vac), axis=0)
        q = p.mean(axis=(1, 2))
        p = np.clip(p, 1e-14, None)
        p /= p.sum(axis=0, keepdims=True)
        q = np.maximum(q, 1e-14)
        q /= q.sum()
        ent = float(-np.sum(q * np.log(q)))
        kl[t] = float(np.mean(np.sum(p * np.log(p / q[:, None, None]), axis=0)))
        norm[t] = kl[t] / ent if ent > 0 else np.nan
    return kl, norm


def _safemean(x):
    a = np.asarray(x, float)
    good = np.isfinite(a)
    return float(a[good].mean()) if np.any(good) else np.nan


def _temporal_scores(lag, band, max_lag_fraction=.75):
    C = np.asarray(lag["C"])
    k = np.asarray(lag["k"])
    tau = np.asarray(lag["tau"])
    counts = np.asarray(lag["counts"])
    selected = (k >= band[0]) & (k <= band[1]) & (counts > 0)
    if not np.any(selected) or len(tau) < 3:
        return np.nan, np.nan
    # Ignore longest lags, which have very few temporal pairs.
    valid_lags = ((tau > 0) &
                  (tau <= max_lag_fraction * float(tau[-1])))
    if not np.any(valid_lags):
        return np.nan, np.nan
    tr0 = np.real(C[0, selected, 0, 0] + C[0, selected, 1, 1])
    normalizer = _safemean(tr0)
    if not np.isfinite(normalizer) or normalizer <= 0:
        return np.nan, np.nan
    asym = C[valid_lags, :, 0, 1][:, selected] - C[valid_lags, :, 1, 0][:, selected]
    with np.errstate(invalid="ignore"):
        asym_score = float(np.sqrt(np.nanmean(np.abs(asym)**2)) / normalizer)
        lag_trace = np.real(C[valid_lags, :, 0, 0][:, selected] +
                            C[valid_lags, :, 1, 1][:, selected])
        negative = max(0., -float(np.nanmin(np.nanmean(lag_trace, axis=1))) / normalizer)
    return asym_score, negative


def _eig_opts(args):
    return dict(EIGEN_KWARGS,
                k_range=tuple(args.eig_k_range),
                min_modes=args.eig_min_modes,
                min_modulus=args.eig_min_modulus)


def _eigen_summary(rec, args, burn):
    from fhd.low_k_recording import eigenvalues_from_modes
    options = _eig_opts(args)
    full = eigenvalues_from_modes(rec, burn_in=burn, **options)
    def n_cycles(e, T):
        w = float(e["Omega_at_r_complex_max"])
        return w * T / (2 * np.pi) if np.isfinite(w) else np.nan
    dt_saved = float(rec["dt_saved"])
    total_frames = rec["mode_history"].shape[1]
    dtobs = max(0., (total_frames - 1 - burn) * dt_saved)
    result = {key: np.asarray(full[key]) for key in (
        "Omega_max", "r_max", "r_at_Omega_max", "k_Omega_max",
        "k_r_max", "r_complex_max", "Omega_at_r_complex_max",
        "k_r_complex_max")}
    result["N_cycles"] = np.float64(n_cycles(full, dtobs))
    result["N_cycles_at_Omega_max"] = np.float64(
        float(full["Omega_max"]) * dtobs / (2*np.pi))
    # Distinguish early linear growth from late nonlinear saturation.
    # Split POST-burn interval into two halves with the same initial burn.
    middle = burn + (total_frames - burn) // 2
    for label, lo, hi in (("early", burn, middle),
                          ("late", middle, total_frames)):
        if hi - lo >= 10:
            try:
                e = eigenvalues_from_modes(rec, burn_in=lo, end_frame=hi,
                                          **options)
                T = (hi - lo - 1) * dt_saved
                result[f"r_complex_{label}"] = np.float64(e["r_complex_max"])
                result[f"Omega_complex_{label}"] = np.float64(e["Omega_at_r_complex_max"])
                result[f"N_cycles_{label}"] = np.float64(n_cycles(e, T))
            except (ValueError, np.linalg.LinAlgError):
                result[f"r_complex_{label}"] = np.float64(np.nan)
                result[f"Omega_complex_{label}"] = np.float64(np.nan)
                result[f"N_cycles_{label}"] = np.float64(np.nan)
    return result


def _analyze_one(job):
    path_str, out_str, args_map, overwrite = job
    path = Path(path_str); out = Path(out_str)
    derived_path = out / path.parent.name / path.name
    if derived_path.exists() and not overwrite:
        return f"SKIP {derived_path}"
    from fhd.low_k_recording import dynamic_from_modes
    from fhd.observables import excess_spectral_power, segregation_index, occupation_index

    with np.load(path, allow_pickle=False) as f:
        rec = {key: np.asarray(f[key]) for key in f.files}
    if rec["mode_history"].shape[0] != 2 or rec["snapshots"].shape[0] != 2:
        raise ValueError(f"Invalid array shapes in {path}")
    # The recording is the single source of truth for parameters and clocks.
    dt = float(rec["dt"])
    L = tuple(map(float, rec["L"]))
    n_cells = tuple(map(int, rec["N"]))
    cap = (L[0] / n_cells[0]) * (L[1] / n_cells[1]) / float(rec["h"])**2
    if not np.isclose(cap, float(rec["people_per_cell"]), rtol=1e-10):
        raise ValueError("Recording h and people-per-cell disagree")
    burn = int(np.ceil(float(args_map["burn_time"]) / float(rec["dt_saved"])))
    if burn >= rec["mode_history"].shape[1] - 2:
        raise ValueError("Burn-in leaves fewer than 3 mode frames")
    # Adapter object for eigenvalue option access, with no argparse coupling.
    class Opt: pass
    opts = Opt()
    for k, v in args_map.items(): setattr(opts, k, v)
    dyn = dynamic_from_modes(
        rec, burn_in=burn,
        n_bins=opts.dynamic_bins, demean_time=False,
        window="hann", mode_chunk_size=128,
    )
    lag = dyn["time_lag"]; spec = dyn["spatiotemporal"]
    eig = _eigen_summary(rec, opts, burn)
    scores = _temporal_scores(lag, tuple(opts.eig_k_range))

    snapshots = np.asarray(rec["snapshots"])
    excess = excess_spectral_power(
        snapshots, L=L, bc="Neumann", reference="beta0",
        reference_cell_capacity=cap,
        reference_rho=np.asarray(snapshots[:, 0]).mean(axis=(-2,-1)),
        n_bins=opts.spatial_bins,
        k_band=tuple(opts.excess_k_band),
    )
    seg = segregation_index(snapshots, cell_capacity=cap,
                            subtract_mixing=True)
    occ = occupation_index(snapshots, reference="initial")
    entropy, normalized_entropy = _entropy_kl_per_frame(snapshots)
    spk = np.asarray(excess["S_plus"], float)
    smk = np.asarray(excess["S_minus"], float)
    nt = len(rec["snapshot_steps"])
    late_start = min(nt - 1, max(0, int(np.floor((1-opts.late_fraction) * nt))))
    late = slice(late_start, nt)
    k_sp = np.asarray(excess["k"], float)
    keep_sp = np.asarray(excess["counts"]) > 0
    def peak_k(arr):
        mean_s = np.full(len(k_sp), np.nan)
        if np.any(keep_sp):
            mean_s[keep_sp] = np.nanmean(arr[late, :][:, keep_sp], axis=0)
        good = keep_sp & np.isfinite(mean_s) & (k_sp > 0)
        return float(k_sp[good][np.argmax(mean_s[good])]) if np.any(good) else np.nan

    output = dict(
        derived_version=np.int64(1),
        family=rec["family"], beta=rec["beta"], x=rec["x"],
        ib=rec["ib"], ix=rec["ix"], rep=rec["rep"],
        seed=rec["seed"], L=rec["L"], N=rec["N"],
        dt=rec["dt"], dt_saved=rec["dt_saved"], h=rec["h"],
        people_per_cell=rec["people_per_cell"],
        reference_cell_capacity=np.float64(cap),
        analysis_settings_json=np.array(json.dumps(args_map, sort_keys=True)),
        eigen_settings_json=np.array(json.dumps(_eig_opts(opts), sort_keys=True)),
        burn_in_mode_frames=np.int64(burn),
        T_observed=np.float64((rec["mode_history"].shape[1]-1-burn)*float(rec["dt_saved"])),
        snapshot_steps=rec["snapshot_steps"],
        snapshot_times=rec["snapshot_times"],
        k_spatial=excess["k"], k_spatial_edges=excess["k_edges"],
        counts_spatial=excess["counts"],
        S_k_t=excess["S"],
        S_plus_k_t=excess["S_plus"], S_minus_k_t=excess["S_minus"],
        excess_plus_k_t=excess["excess_plus"],
        excess_minus_k_t=excess["excess_minus"],
        P_plus=excess["P_plus"], P_minus=excess["P_minus"],
        C=excess["C"], order_threshold=np.float64(excess["order_threshold"]),
        I_seg=seg["I_seg"], I_seg_excess=seg["excess"],
        I_occ=occ["I_occ"], entropy_kl=entropy,
        normalized_entropy_segregation=normalized_entropy,
        temporal_antisymmetry=np.float64(scores[0]),
        oscillation_negative_lobe=np.float64(scores[1]),
        k_dynamic=lag["k"], tau=lag["tau"], omega=spec["omega"],
        counts_dynamic=lag["counts"],
        demean_time=np.bool_(False),
    )
    if not opts.no_full_dynamics:
        output.update(C_k_tau=lag["C"], S_k_omega=spec["S"])
    output.update(eig)
    for name, values in (
        ("C_late", excess["C"]),
        ("P_plus_late", excess["P_plus"]),
        ("P_minus_late", excess["P_minus"]),
        ("I_seg_late", seg["I_seg"]),
        ("I_seg_excess_late", seg["excess"]),
        ("I_occ_late", occ["I_occ"]),
        ("entropy_kl_late", entropy),
        ("normalized_entropy_segregation_late", normalized_entropy),
    ):
        output[name] = np.float64(_safemean(np.asarray(values)[late]))
    output["k_peak_plus_late"] = np.float64(peak_k(spk))
    output["k_peak_minus_late"] = np.float64(peak_k(smk))
    _atomic_npz(derived_path, output)
    return f"OK analysis {path.name}: r={output['r_max']:.4g}, r_complex={output['r_complex_max']:.4g}, Ncycles={output['N_cycles']:.2f}"


def _jobs(paths, worker, max_workers):
    if not paths: return
    if max_workers == 1:
        for p in paths:
            print(worker(p), flush=True)
    else:
        # Spawn is safest with Numba / OpenMP in a long-lived Python process.
        from multiprocessing import get_context
        with ProcessPoolExecutor(max_workers=max_workers,
                                 mp_context=get_context("spawn")) as ex:
            futures = {ex.submit(worker, j): j for j in paths}
            for fut in as_completed(futures):
                print(fut.result(), flush=True)


def _simulate(args):
    _check_args(args)
    cfg = _grid_config(args)
    _write_config(args.out, cfg)
    jobs = []
    for family in cfg["families"]:
        xs = _xs_for_family(cfg, family)
        for ib in range(len(cfg["beta"])):
            for ix in range(len(xs)):
                for rep in range(args.repeats):
                    path = _run_path(args.out, family, ib, ix, rep)
                    if not path.exists() or args.overwrite:
                        jobs.append((cfg, str(args.out), family, ib, ix, rep, args.overwrite))
    if args.limit is not None: jobs = jobs[:args.limit]
    print(f"Simulating {len(jobs)} jobs using {args.workers} workers.")
    _jobs(jobs, _simulate_one, args.workers)


def _analyze(args):
    cfg = _load_config(args.out)
    paths = sorted(p for family in cfg["families"]
                   for p in (args.out / "recordings" / family).glob("b*_x*_r*.npz"))
    if not args.overwrite:
        paths = [p for p in paths if not (args.out / p.parent.name / p.name).exists()]
    if args.limit is not None: paths = paths[:args.limit]
    keys = ("burn_time", "late_fraction", "spatial_bins", "dynamic_bins",
            "excess_k_band", "eig_k_range", "eig_min_modes",
            "eig_min_modulus", "no_full_dynamics")
    options = {k: getattr(args, k) for k in keys}
    jobs = [(str(p), str(args.out), options, args.overwrite) for p in paths]
    print(f"Analyzing {len(jobs)} recordings with {args.workers} workers.")
    _jobs(jobs, _analyze_one, args.workers)


def _aggregate(args):
    cfg = _load_config(args.out)
    betas = np.asarray(cfg["beta"], float)
    for family in cfg["families"]:
        xs = _xs_for_family(cfg, family)
        sums = {k: np.zeros((len(betas), len(xs))) for k in GRID_SCALARS}
        counts = {k: np.zeros((len(betas), len(xs)), dtype=int) for k in GRID_SCALARS}
        n_runs = np.zeros((len(betas), len(xs)), dtype=int)
        for path in sorted((args.out / family).glob("b*_x*_r*.npz")):
            match = REC_NAME.fullmatch(path.name)
            if match is None: continue
            ib, ix, rep = map(int, match.groups())
            if ib >= len(betas) or ix >= len(xs): continue
            with np.load(path, allow_pickle=False) as data:
                if not (np.isclose(float(data["beta"]), betas[ib]) and
                        np.isclose(float(data["x"]), xs[ix])):
                    warnings.warn(f"Skipping mismatched {path}")
                    continue
                n_runs[ib, ix] += 1
                for k in GRID_SCALARS:
                    if k not in data: continue
                    val = float(data[k])
                    if np.isfinite(val):
                        sums[k][ib, ix] += val
                        counts[k][ib, ix] += 1
        payload = dict(beta=betas, x=xs, n_runs=n_runs)
        for k in GRID_SCALARS:
            val = np.full((len(betas), len(xs)), np.nan)
            np.divide(sums[k], counts[k], out=val, where=counts[k] > 0)
            payload[k] = val
            payload[k + "_n"] = counts[k]
        _atomic_npz(args.out / f"{family}_grids.npz", payload)
        print(f"{family}: aggregated {n_runs.sum()} analyzed runs; "
              f"{np.count_nonzero(n_runs)} / {n_runs.size} grid points")


def main(argv=None):
    args = parse_args(argv)
    if args.stage in ("simulate", "all"):
        _simulate(args)
    if args.stage in ("analyze", "all"):
        _analyze(args)
    if args.stage in ("aggregate", "all"):
        _aggregate(args)


if __name__ == "__main__":
    main()
