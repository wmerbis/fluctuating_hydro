#!/usr/bin/env python3
"""
Final benchmark: Gaussian finite-volume Schelling noise vs exact face reactions.

Compares schelling_flux='finite_volume' and 'face_reaction' under matched
initial conditions and seeds in three scenarios:

1) schelling_segregating: long pure-Schelling coarsening run.
2) strong_voter: same Schelling parameters with D_v ~ D_a.
3) theory_spectrum: stable homogeneous case for linear-theory comparison.

Outputs per-run NPZ/JSON files, summary CSV tables and PNG figures.
"""
from __future__ import annotations

import csv
import importlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
# Configuration
# =============================================================================

OUTPUT_DIR = Path('benchmark_face_reaction_vs_fv')
RUN_DIR = OUTPUT_DIR / 'runs'
FIG_DIR = OUTPUT_DIR / 'figures'

METHODS = ('finite_volume', 'face_reaction')

N = (128, 128)
L = (50.0, 50.0)
DT = 1.0e-3
RHO_A0 = 0.35
RHO_B0 = 0.35
# OMEGA = 10_000.0
DX = L[0] / N[0]
DY = L[1] / N[1]
H = 0.01

D = np.array([0.1, 0.1])
BETA = 10.0
GAMMA = np.eye(2)
KAPPA_SEGREGATING = np.array([[0.6, -0.4], [-0.4, 0.6]])

# Deliberately stable homogeneous case for the linear spectrum comparison.
# Replace by the stable parameter set used in the paper's spectrum figure if desired.
KAPPA_THEORY = np.zeros((2, 2))

N_RUNS = 6
INITIAL_PERTURBATION = 1.0e-3
INIT_SEED_BASE = 31_000
SIM_SEED_BASE = 71_000

USE_FASTPATH = False
PROJECTION_MODE = 'redistribute'
USE_NUMBA_PROJECTION = True
NUMBA_PROJECTION_THREADS = 1
VOTER_NOISE_MODE = 'wright_fisher'
WF_GAUSSIAN_THRESHOLD = 0.01
ENABLE_REACTION_DIAGNOSTICS = True
RESUME = True

STRUCTURE_FACTOR_MODULE = 'structure_factor'
ENABLE_THEORY = True


@dataclass(frozen=True)
class Scenario:
    name: str
    model: str
    nsteps: int
    nframes: int
    D_v: float
    kappa: np.ndarray
    spectrum_tail_fraction: float
    spectrum_stride: int
    compare_theory: bool = False

    @property
    def save_every(self):
        if self.nsteps % self.nframes != 0:
            raise ValueError(f'{self.name}: nsteps must be divisible by nframes')
        return self.nsteps // self.nframes


SCENARIOS = (
    Scenario('schelling_segregating', 'Vitelli', 100_000, 200, 0.0,
             KAPPA_SEGREGATING, 0.25, 4, False),
    Scenario('strong_voter', 'Schelling+Voter', 50_000, 200, 0.1,
             KAPPA_SEGREGATING, 0.50, 2, False),
    Scenario('theory_spectrum', 'Schelling+Voter', 30_000, 150, 0.1,
             KAPPA_THEORY, 0.50, 1, True),
)


# =============================================================================
# Initial condition / simulator
# =============================================================================

def make_initial_condition(seed):
    """Small zero-mean perturbation around rho_A=rho_B=0.35."""
    rng = np.random.default_rng(seed)
    dA = rng.normal(0.0, INITIAL_PERTURBATION, size=N)
    dB = rng.normal(0.0, INITIAL_PERTURBATION, size=N)
    dA -= dA.mean()
    dB -= dB.mean()

    phi = np.empty((2,) + N, dtype=np.float64)
    phi[0] = RHO_A0 + dA
    phi[1] = RHO_B0 + dB

    if phi.min() <= 0.0 or phi.sum(axis=0).max() >= 1.0:
        raise RuntimeError('Initial perturbation left the physical simplex')
    return phi


def make_param(scenario):
    return {
        'D': D.copy(),
        'D_v': float(scenario.D_v),
        'beta': float(BETA),
        'kappa': scenario.kappa.copy(),
        'Gamma': GAMMA.copy(),
        'h': float(H),
        'noise_v': 1.0,
    }


def make_sim(method):
    sim = fhd.fhd_2d(
        L, N, bc='Neumann', fft=False,
        schelling_flux=method,
        projection_mode=PROJECTION_MODE,
        use_numba_projection=USE_NUMBA_PROJECTION,
        numba_projection_threads=NUMBA_PROJECTION_THREADS,
        voter_noise_mode=VOTER_NOISE_MODE,
        wf_gaussian_threshold=WF_GAUSSIAN_THRESHOLD,
    )

    # Development versions used slightly different names for this switch.
    if ENABLE_REACTION_DIAGNOSTICS:
        for name in ('reaction_diagnostics', 'schelling_reaction_diagnostics',
                     'enable_reaction_diagnostics'):
            if hasattr(sim, name):
                setattr(sim, name, True)
    return sim


def warmup_sim(sim, phi):
    """Keep Numba projection compilation outside the timed region."""
    work = sim._ensure_work(phi.dtype) if hasattr(sim, '_ensure_work') else None
    if hasattr(sim, '_warmup_projection_kernels'):
        sim._warmup_projection_kernels(phi, work=work)


# =============================================================================
# DCT structure factors and coarsening
# =============================================================================

kx = np.pi * np.arange(N[0], dtype=float) / L[0]
ky = np.pi * np.arange(N[1], dtype=float) / L[1]
KX, KY = np.meshgrid(kx, ky, indexing='ij')
KMAG = np.sqrt(KX*KX + KY*KY)
NCELLS = N[0] * N[1]


def dct_modal_spectra(phi):
    """
    DCT-II structure factors for the cell-centered Neumann modes.

    Division by Ncells makes sum_k S_aa(k) equal the spatial variance of rho_a.
    """
    dA = phi[0] - phi[0].mean()
    dB = phi[1] - phi[1].mean()
    Ahat = dctn(dA, type=2, norm='ortho')
    Bhat = dctn(dB, type=2, norm='ortho')
    Saa = Ahat*Ahat / NCELLS
    Sbb = Bhat*Bhat / NCELLS
    Sab = Ahat*Bhat / NCELLS
    return {
        'Saa': Saa,
        'Sbb': Sbb,
        'Sab': Sab,
        'Spol': Saa + Sbb - 2.0*Sab,
        'Socc': Saa + Sbb + 2.0*Sab,
    }


def characteristic_k(phi):
    """<k> of the polarization spectrum and ell=2*pi/<k>."""
    pol = phi[0] - phi[1]
    pol -= pol.mean()
    phat = dctn(pol, type=2, norm='ortho')
    power = phat*phat
    power[0, 0] = 0.0
    denom = power.sum()
    if denom <= 0.0:
        return np.nan, np.nan
    kmean = float(np.sum(KMAG*power) / denom)
    ell = 2.0*np.pi/kmean if kmean > 0.0 else np.inf
    return kmean, ell


def radial_average(power, nbins=90):
    k = KMAG.ravel()
    p = power.ravel()
    edges = np.linspace(0.0, float(k.max()), nbins + 1)
    which = np.digitize(k, edges) - 1
    kval, pval, counts = [], [], []
    for b in range(nbins):
        mask = which == b
        n = int(np.count_nonzero(mask))
        if n:
            kval.append(float(k[mask].mean()))
            pval.append(float(p[mask].mean()))
            counts.append(n)
    return np.asarray(kval), np.asarray(pval), np.asarray(counts)


def trajectory_observables(phi_run, dt_frame):
    nf = phi_run.shape[1]
    keys = ('mean_A', 'mean_B', 'mean_0', 'var_A', 'var_B', 'var_0',
            'var_pol', 'var_occ', 'dissimilarity', 'relative_entropy',
            'k_mean_pol', 'ell_pol')
    out = {'time': np.arange(nf, dtype=float) * dt_frame}
    out.update({k: np.empty(nf) for k in keys})

    for j in range(nf):
        phi = phi_run[:, j]
        A, B = phi
        vac = 1.0 - A - B
        pol = A - B
        occ = A + B

        out['mean_A'][j] = A.mean()
        out['mean_B'][j] = B.mean()
        out['mean_0'][j] = vac.mean()
        out['var_A'][j] = A.var()
        out['var_B'][j] = B.var()
        out['var_0'][j] = vac.var()
        out['var_pol'][j] = pol.var()
        out['var_occ'][j] = occ.var()
        out['dissimilarity'][j] = dissimilarity(phi)
        out['relative_entropy'][j] = mean_relative_entropy(phi)
        out['k_mean_pol'][j], out['ell_pol'][j] = characteristic_k(phi)
    return out


def late_time_spectrum(phi_run, scenario):
    """Average DCT spectra over the selected late-time saved frames."""
    nf = phi_run.shape[1]
    first = max(1, int((1.0 - scenario.spectrum_tail_fraction) * nf))
    indices = np.arange(first, nf, scenario.spectrum_stride, dtype=int)
    accum = None

    for j in indices:
        s = dct_modal_spectra(phi_run[:, j])
        if accum is None:
            accum = {k: v.copy() for k, v in s.items()}
        else:
            for k in accum:
                accum[k] += s[k]

    for k in accum:
        accum[k] /= len(indices)

    out = {'spectrum_frame_indices': indices}
    for key, power in accum.items():
        kval, radial, counts = radial_average(power)
        out[f'k_{key}'] = kval
        out[f'radial_{key}'] = radial
        out[f'counts_{key}'] = counts
    return out


# =============================================================================
# Projection / reaction diagnostics
# =============================================================================

def scalar_diagnostics(sim):
    """Exact final scalar diagnostics accumulated over the entire run."""
    if not hasattr(sim, '_ensure_work'):
        return {}
    diag = sim._ensure_work().get('projection_diag', {})
    out = {}
    for key, value in diag.items():
        if isinstance(value, (bool, int, float, np.integer, np.floating)):
            out[key] = value.item() if isinstance(value, np.generic) else value
    return out


def sampled_projection_history(sim, dt):
    """
    Sampled per-call projection activity.

    Exact integrated totals are taken separately from scalar_diagnostics().
    """
    if not hasattr(sim, '_ensure_work'):
        return {}
    hist = sim._ensure_work().get('projection_diag', {}).get('history', [])
    if not hist:
        return {}

    fields = ('n_low_entries', 'n_high_entries', 'n_simplex_cells',
              'mass_added_low', 'mass_removed_high', 'mass_removed_simplex',
              'net_change', 'needs_projection')
    out = {
        'proj_step': np.array([h.get('step', -1) for h in hist], dtype=int),
        'proj_time': np.array([h.get('step', -1)*dt for h in hist], dtype=float),
        'proj_stage': np.array([str(h.get('stage', 'unknown')) for h in hist], dtype='U32'),
    }
    for field in fields:
        out[f'proj_{field}'] = np.array([h.get(field, 0) for h in hist], dtype=float)
    return out


# =============================================================================
# Theory adapter
# =============================================================================
from fhd.structure_factor import structure_factor


def theory_structure_factor(k_bins, rho_A, rho_B, param):
    """
    Evaluate the analytical equal-time structure factor on the supplied
    one-dimensional k grid.

    Parameters
    ----------
    k_bins : ndarray
        Wavenumbers at which the theoretical structure factor is evaluated.

    rho_A, rho_B : float
        Homogeneous background densities used in the linear theory. For a
        stationary numerical trajectory these can be taken as late-time
        space-time averages.

    param : dict
        Model parameter dictionary passed to fhd.structure_factor.structure_factor.

    Returns
    -------
    dict
        Arrays Saa, Sab, Sbb and the derived polarization and occupancy spectra.
    """
    S_AA = np.zeros(len(k_bins))
    S_AB = np.zeros(len(k_bins))
    S_BB = np.zeros(len(k_bins))

    for i, k in enumerate(k_bins):
        S = structure_factor(k, rho_A, rho_B, param)
        S_AA[i] = S[0, 0]
        S_AB[i] = S[0, 1]
        S_BB[i] = S[1, 1]

    return {
        "Saa": S_AA,
        "Sab": -S_AB,
        "Sbb": S_BB,
        "Spol": S_AA + S_BB - 2.0 * S_AB,
        "Socc": S_AA + S_BB + 2.0 * S_AB,
    }


def fit_log_amplitude(data, theory, valid=None):
    """
    Fit one multiplicative normalization factor between data and theory by
    minimizing their mean difference in log space.

        scale = exp(<log(data) - log(theory)>)

    Only finite, strictly positive entries are used.
    """
    if valid is None:
        valid = (
            np.isfinite(data)
            & np.isfinite(theory)
            & (data > 0.0)
            & (theory > 0.0)
        )
    else:
        valid = (
            valid
            & np.isfinite(data)
            & np.isfinite(theory)
            & (data > 0.0)
            & (theory > 0.0)
        )

    if np.count_nonzero(valid) == 0:
        return np.nan

    log_amp = np.mean(np.log(data[valid]) - np.log(theory[valid]))
    return float(np.exp(log_amp))


def theory_scores(k, data, theory, kmin=None, kmax=None):
    """
    Compare an empirical spectrum with theory after fitting the overall
    multiplicative normalization in log space.

    Returns both the fitted normalization and shape-error measures.
    """
    valid = (
        np.isfinite(k)
        & np.isfinite(data)
        & np.isfinite(theory)
        & (k > 0.0)
        & (data > 0.0)
        & (theory > 0.0)
    )

    if kmin is not None:
        valid &= k >= kmin
    if kmax is not None:
        valid &= k <= kmax

    if np.count_nonzero(valid) < 3:
        return {
            "n_modes": int(np.count_nonzero(valid)),
            "scale": np.nan,
            "log_rmse": np.nan,
            "relative_rmse": np.nan,
        }

    scale = fit_log_amplitude(data, theory, valid=valid)
    theory_scaled = scale * theory

    log_rmse = np.sqrt(
        np.mean(
            (
                np.log(data[valid])
                - np.log(theory_scaled[valid])
            ) ** 2
        )
    )

    relative_rmse = np.sqrt(
        np.mean(
            (
                data[valid]
                - theory_scaled[valid]
            ) ** 2
        )
    ) / np.sqrt(
        np.mean(data[valid] ** 2)
    )

    return {
        "n_modes": int(np.count_nonzero(valid)),
        "scale": float(scale),
        "log_rmse": float(log_rmse),
        "relative_rmse": float(relative_rmse),
    }

# =============================================================================
# Run one simulation
# =============================================================================

def run_path(scenario, method, run_index):
    return RUN_DIR / f'{scenario.name}__{method}__run{run_index:02d}.npz'


def json_path(scenario, method, run_index):
    return RUN_DIR / f'{scenario.name}__{method}__run{run_index:02d}.json'


def run_one(scenario, method, run_index):
    npz_path = run_path(scenario, method, run_index)
    js_path = json_path(scenario, method, run_index)
    if RESUME and npz_path.exists() and js_path.exists():
        print(f'  SKIP existing: {npz_path.name}', flush=True)
        return

    init_seed = INIT_SEED_BASE + run_index
    sim_seed = SIM_SEED_BASE + run_index
    phi_init = make_initial_condition(init_seed)
    param = make_param(scenario)

    sim = make_sim(method)
    sim.set_seed(sim_seed)
    np.random.seed(sim_seed)  # legacy Gaussian paths may still use np.random
    warmup_sim(sim, phi_init)

    print(f'  {method:16s} run={run_index:02d} steps={scenario.nsteps}', flush=True)
    t0 = time.perf_counter()
    phi_run = sim.run(
        phi_init, param, scenario.nsteps, DT, True, scenario.nframes,
        scheme='FE', model=scenario.model, verbatum=False,
        diagnostic_interval=scenario.save_every,
        reset_projection_diag=True, use_fastpath=USE_FASTPATH,
    )
    runtime = time.perf_counter() - t0

    obs = trajectory_observables(phi_run, scenario.save_every*DT)
    spectra = late_time_spectrum(phi_run, scenario)
    proj_hist = sampled_projection_history(sim, DT)
    diag = scalar_diagnostics(sim)

    arrays = {}
    arrays.update(obs)
    arrays.update(spectra)
    arrays.update(proj_hist)
    arrays['phi_final'] = phi_run[:, -1]
    arrays['runtime'] = np.array(runtime)
    arrays['init_seed'] = np.array(init_seed)
    arrays['sim_seed'] = np.array(sim_seed)
    np.savez_compressed(npz_path, **arrays)

    meta = {
        'scenario': scenario.name,
        'method': method,
        'run_index': run_index,
        'runtime': runtime,
        'N': list(N), 'L': list(L), 'dt': DT,
        'nsteps': scenario.nsteps, 'nframes': scenario.nframes,
        'D': D.tolist(), 'D_v': scenario.D_v, 'beta': BETA,
        'kappa': scenario.kappa.tolist(), 'Gamma': GAMMA.tolist(),
        'h': H,
        'projection_diagnostics': diag,
    }
    with open(js_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"    runtime={runtime:.1f}s  D={obs['dissimilarity'][-1]:.4f}  "
          f"<k>={obs['k_mean_pol'][-1]:.4f}  "
          f"low={diag.get('n_low_entries', 0)}  "
          f"simplex={diag.get('n_simplex_cells', 0)}", flush=True)


# =============================================================================
# Ensemble helpers / plots
# =============================================================================

def load_run(scenario, method, run_index):
    arr = dict(np.load(run_path(scenario, method, run_index), allow_pickle=False))
    with open(json_path(scenario, method, run_index)) as f:
        meta = json.load(f)
    return arr, meta


def ensemble_arrays(scenario, method, key):
    return np.asarray([load_run(scenario, method, r)[0][key] for r in range(N_RUNS)])


def mean_sem(x, axis=0):
    mean = np.nanmean(x, axis=axis)
    sem = (np.nanstd(x, axis=axis, ddof=1) / math.sqrt(x.shape[axis])
           if x.shape[axis] > 1 else np.full_like(mean, np.nan, dtype=float))
    return mean, sem


def plot_timeseries(scenario, key, ylabel):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method in METHODS:
        t = ensemble_arrays(scenario, method, 'time')[0]
        y = ensemble_arrays(scenario, method, key)
        mean, sem = mean_sem(y)
        ax.plot(t, mean, label=method)
        ax.fill_between(t, mean-sem, mean+sem, alpha=0.2)
    ax.set_xlabel('time')
    ax.set_ylabel(ylabel)
    ax.set_title(scenario.name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f'{scenario.name}__{key}.png', dpi=200)
    plt.close(fig)


def plot_projection_history(scenario):
    """Sampled Schelling-stage low-density projection entries over time."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method in METHODS:
        series = []
        for r in range(N_RUNS):
            arr, _ = load_run(scenario, method, r)
            if 'proj_time' not in arr:
                continue
            stage = arr['proj_stage']
            mask = stage == 'schelling'
            if not np.any(mask):
                mask = np.ones(stage.shape, dtype=bool)
            t = arr['proj_time'][mask]
            y = arr['proj_n_low_entries'][mask]
            ut = np.unique(t)
            ys = np.array([y[t == tt].sum() for tt in ut])
            series.append((ut, ys))
        if not series:
            continue
        common_t = series[0][0]
        vals = np.asarray([y for t, y in series if np.allclose(t, common_t)])
        mean, sem = mean_sem(vals)
        ax.plot(common_t, mean, label=method)
        ax.fill_between(common_t, mean-sem, mean+sem, alpha=0.2)
    ax.set_xlabel('time')
    ax.set_ylabel('sampled low-density entries / projection call')
    ax.set_title(f'{scenario.name}: projection activity')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f'{scenario.name}__projection_history.png', dpi=200)
    plt.close(fig)


def plot_coarsening(scenario):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method in METHODS:
        t = ensemble_arrays(scenario, method, 'time')[0]
        y = ensemble_arrays(scenario, method, 'k_mean_pol')
        mean, sem = mean_sem(y)
        ax.plot(t, mean, label=method)
        ax.fill_between(t, mean-sem, mean+sem, alpha=0.2)
    ax.set_xlabel('time')
    ax.set_ylabel(r'$\langle k\rangle_{\rm pol}$')
    ax.set_title(f'{scenario.name}: coarsening')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f'{scenario.name}__kmean.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method in METHODS:
        t = ensemble_arrays(scenario, method, 'time')[0]
        y = ensemble_arrays(scenario, method, 'ell_pol')
        mean, _ = mean_sem(y)
        mask = (t > 0.0) & np.isfinite(mean) & (mean > 0.0)
        ax.loglog(t[mask], mean[mask], label=method)
    ax.set_xlabel('time')
    ax.set_ylabel(r'$\ell_{\rm pol}=2\pi/\langle k\rangle$')
    ax.set_title(f'{scenario.name}: characteristic length')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f'{scenario.name}__ell_loglog.png', dpi=200)
    plt.close(fig)


def ensemble_radial_spectrum(scenario, method, field):
    k_all = ensemble_arrays(scenario, method, f'k_{field}')
    s_all = ensemble_arrays(scenario, method, f'radial_{field}')
    k = k_all[0]
    if not np.allclose(k_all, k[None, :], equal_nan=True):
        raise RuntimeError('Radial k grids differ across runs')
    mean, sem = mean_sem(s_all)
    return k, mean, sem


def plot_empirical_spectra(scenario):
    for field, ylabel in (('Saa', r'$S_{AA}(k)$'), ('Spol', r'$S_{pol}(k)$'),
                           ('Socc', r'$S_{occ}(k)$')):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for method in METHODS:
            k, mean, _ = ensemble_radial_spectrum(scenario, method, field)
            mask = (k > 0.0) & np.isfinite(mean) & (mean > 0.0)
            ax.loglog(k[mask], mean[mask], label=method)
        ax.set_xlabel(r'$k$')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{scenario.name}: late-time DCT spectrum')
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / f'{scenario.name}__{field}_spectrum.png', dpi=200)
        plt.close(fig)


def fit_coarsening_exponent(scenario, method):
    """Diagnostic late-time fit ell ~ t^alpha; finite-size saturation can bias it."""
    t = ensemble_arrays(scenario, method, 'time')[0]
    ell = ensemble_arrays(scenario, method, 'ell_pol')
    mean = np.nanmean(ell, axis=0)
    first = len(t)//2
    mask = ((np.arange(len(t)) >= first) & (t > 0.0) &
            np.isfinite(mean) & (mean > 0.0))
    if np.count_nonzero(mask) < 5:
        return np.nan
    alpha, _ = np.polyfit(np.log(t[mask]), np.log(mean[mask]), 1)
    return float(alpha)


# =============================================================================
# Theory comparison
# =============================================================================

def compare_with_theory(scenario):
    if not ENABLE_THEORY or not scenario.compare_theory:
        return []

    param = make_param(scenario)
    rho_bg = np.array([RHO_A0, RHO_B0], dtype=float)
    rows = []

    for method in METHODS:
        k, Saa, _ = ensemble_radial_spectrum(scenario, method, 'Saa')
        _, Sbb, _ = ensemble_radial_spectrum(scenario, method, 'Sbb')
        _, Sab, _ = ensemble_radial_spectrum(scenario, method, 'Sab')

        try:
            theory = theory_structure_factor(k, rho_bg[0], rho_bg[1], param)
        except Exception as exc:
            print('\nTHEORY ADAPTER WARNING\n----------------------')
            print(exc)
            print('Empirical benchmark completed; edit theory_structure_factor() for your API.')
            return []

        # theory = {
        #     'Saa': G[:, 0, 0],
        #     'Sbb': G[:, 1, 1],
        #     'Sab': - G[:, 0, 1],
        #     'Spol': G[:, 0, 0] + G[:, 1, 1] - 2.0*G[:, 0, 1],
        #     'Socc': G[:, 0, 0] + G[:, 1, 1] + 2.0*G[:, 0, 1],
        # }
        empirical = {
            'Saa': Saa, 'Sbb': Sbb, 'Sab': - Sab,
            'Spol': Saa + Sbb - 2.0*Sab,
            'Socc': Saa + Sbb + 2.0*Sab,
        }

        for field in ('Saa', 'Sbb', 'Spol', 'Socc'):
            rows.append({'scenario': scenario.name, 'method': method,
                         'field': field,
                         **theory_scores(k, empirical[field], theory[field], kmin=1, kmax=8)})

        for field, ylabel in (('Saa', r'$S_{AA}(k)$'), ('Spol', r'$S_{pol}(k)$')):
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            e, t = empirical[field], theory[field]
            me = (k > 0.0) & np.isfinite(e) & (e > 0.0)
            mt = (k > 0.0) & np.isfinite(t) & (t > 0.0)
            ax.loglog(k[me], e[me], 'o', ms=4, label=f'{method}: simulation')
            ax.loglog(k[mt], t[mt], '-', label='theory')
            ax.set_xlabel(r'$k$')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{scenario.name}: {method}')
            ax.legend()
            fig.tight_layout()
            fig.savefig(FIG_DIR / f'{scenario.name}__{method}__{field}__theory.png', dpi=200)
            plt.close(fig)

    return rows


# =============================================================================
# Summary tables
# =============================================================================

SUMMARY_KEYS = ('dissimilarity', 'relative_entropy', 'var_A', 'var_B', 'var_0',
                'var_pol', 'var_occ', 'k_mean_pol', 'ell_pol')


def write_summary():
    rows = []
    for scenario in SCENARIOS:
        for method in METHODS:
            metas = [load_run(scenario, method, r)[1] for r in range(N_RUNS)]
            runtimes = [m['runtime'] for m in metas]
            diag_rows = [m.get('projection_diagnostics', {}) for m in metas]

            row = {
                'scenario': scenario.name,
                'method': method,
                'runtime_mean_s': float(np.mean(runtimes)),
                'runtime_sd_s': float(np.std(runtimes, ddof=1)) if len(runtimes) > 1 else np.nan,
                'coarsening_alpha': fit_coarsening_exponent(scenario, method),
            }

            for key in SUMMARY_KEYS:
                x = ensemble_arrays(scenario, method, key)[:, -1]
                row[f'{key}_final_mean'] = float(np.nanmean(x))
                row[f'{key}_final_sem'] = (float(np.nanstd(x, ddof=1)/math.sqrt(len(x)))
                                               if len(x) > 1 else np.nan)

            for key in (
                'n_calls', 'n_low_entries', 'n_high_entries', 'n_simplex_cells',
                'mass_added_low', 'mass_removed_high', 'mass_removed_simplex',
                'n_expensive_projection_calls', 'n_roundoff_cleanup_calls',
                'n_schelling_reaction_calls', 'n_schelling_candidates',
                'n_schelling_events', 'max_schelling_candidates',
                'n_schelling_M0_faces', 'n_schelling_M1_faces',
                'n_schelling_Mge2_faces', 'n_schelling_absorbed_source',
                'n_schelling_absorbed_vacancy'):
                row[f'{key}_mean'] = float(np.mean([d.get(key, 0.0) for d in diag_rows]))
            rows.append(row)

    fields = sorted({k for row in rows for k in row})
    with open(OUTPUT_DIR / 'summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print('\nSUMMARY\n=======')
    for row in rows:
        print(f"{row['scenario']:22s} {row['method']:16s} "
              f"runtime={row['runtime_mean_s']:.1f}s  "
              f"D={row['dissimilarity_final_mean']:.4f}  "
              f"<k>={row['k_mean_pol_final_mean']:.4f}  "
              f"low={row['n_low_entries_mean']:.1f}")


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print('='*80)
    print('FINITE VOLUME vs FACE REACTION FINAL BENCHMARK')
    print('='*80)
    print(f'N={N}, L={L}, dt={DT}, D={D}, beta={BETA}, h={H}')
    print(f'N runs={N_RUNS}')

    for scenario in SCENARIOS:
        print('\n' + '='*80)
        print(f'SCENARIO: {scenario.name}')
        print('='*80)
        for r in range(N_RUNS):
            print(f'\nMatched run {r+1}/{N_RUNS}')
            # Alternate order to reduce systematic cache/order bias.
            order = METHODS if r % 2 == 0 else METHODS[::-1]
            for method in order:
                run_one(scenario, method, r)

    for scenario in SCENARIOS:
        plot_timeseries(scenario, 'dissimilarity', 'dissimilarity')
        plot_timeseries(scenario, 'relative_entropy', 'relative entropy')
        plot_timeseries(scenario, 'var_pol', r'$\mathrm{Var}(\rho_A-\rho_B)$')
        plot_timeseries(scenario, 'var_occ', r'$\mathrm{Var}(\rho_A+\rho_B)$')
        plot_projection_history(scenario)
        plot_coarsening(scenario)
        plot_empirical_spectra(scenario)

    write_summary()

    theory_rows = []
    for scenario in SCENARIOS:
        theory_rows.extend(compare_with_theory(scenario))

    if theory_rows:
        fields = sorted({k for row in theory_rows for k in row})
        with open(OUTPUT_DIR / 'theory_scores.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(theory_rows)

        print('\nTHEORY SCORES\n=============')
        for row in theory_rows:
            print(
                f"{row['method']:16s} {row['field']:5s} "
                f"scale={row['scale']:.4g}  "
                f"rel_RMSE={row['relative_rmse']:.4g}  "
                f"log_RMSE={row['log_rmse']:.4g}"
            )

    print(f'\nResults written to: {OUTPUT_DIR.resolve()}')


if __name__ == '__main__':
    main()
