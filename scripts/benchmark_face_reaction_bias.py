"""
benchmark_face_reaction_bias.py

Compare:

    schelling_flux="face_reaction"
        exact passive face reactions + FE utility flux

against

    schelling_flux="face_reaction_biased"
        utility bias incorporated directly into reaction rates

using an ensemble of stochastic Schelling+Voter simulations.
"""

from pathlib import Path
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fhd.FHD_2D import fhd_2d
from fhd.operations import dissimilarity, mean_relative_entropy


# ============================================================
# SETTINGS
# ============================================================

N_RUNS = 8                  # Start small; increase to 20-50 later
N_STEPS = 20_000
DT = 1e-3
NO_FRAMES = 100

N = (128, 128)
L = (128.0, 128.0)
BC = "Neumann"

D = np.array([0.1, 0.1])
D_V = 0.1
BETA = 10.0

KAPPA = np.array([
    [ 0.6, -0.4],
    [-0.4,  0.6],
])

GAMMA = np.eye(2)

RHO_A0 = 0.35
RHO_B0 = 0.35

OMEGA = 10_000
DX = L[0] / N[0]
DY = L[1] / N[1]
H = np.sqrt(DX * DY / OMEGA)

BASE_SEED = 12345

MODES = (
    "face_reaction",
    "face_reaction_biased",
)


# ============================================================
# PARAMETERS
# ============================================================

PARAM = {
    "D": D,
    "D_v": D_V,
    "beta": BETA,
    "kappa": KAPPA,
    "Gamma": GAMMA,
    "h": H,
    "noise_v": 1.0,
}


# ============================================================
# INITIAL CONDITION
# ============================================================

def make_initial_condition():
    phi = np.empty((2,) + N, dtype=float)
    phi[0] = RHO_A0
    phi[1] = RHO_B0
    return phi


# ============================================================
# OBSERVABLES
# ============================================================

OBS_NAMES = (
    "mean_A",
    "mean_B",
    "mean_0",
    "var_A",
    "var_B",
    "var_0",
    "var_pol",
    "var_occ",
    "dissimilarity",
    "relative_entropy",
)


def measure_frame(phi):
    A = phi[0]
    B = phi[1]
    rho0 = 1.0 - A - B
    pol = A - B
    occ = A + B

    return np.array([
        A.mean(),
        B.mean(),
        rho0.mean(),
        A.var(),
        B.var(),
        rho0.var(),
        pol.var(),
        occ.var(),
        dissimilarity(phi),
        mean_relative_entropy(phi),
    ])


def measure_run(phi_run):
    nframes = phi_run.shape[1]
    obs = np.empty((nframes, len(OBS_NAMES)))

    for t in range(nframes):
        obs[t] = measure_frame(phi_run[:, t])

    return obs


# ============================================================
# SIMULATOR
# ============================================================

def make_sim(mode):
    return fhd_2d(
        L=L,
        N=N,
        bc=BC,
        fft=False,
        schelling_flux=mode,
        projection_floor=0.0,
        projection_mode="redistribute",
        voter_noise_mode="wright_fisher",
        wf_gaussian_threshold=0.025,
    )


# ============================================================
# ONE RUN
# ============================================================

def run_one(mode, seed):
    sim = make_sim(mode)
    sim.set_seed(seed)

    phi0 = make_initial_condition()

    t0 = time.perf_counter()

    phi_run = sim.run(
        phi0,
        PARAM,
        nsteps=N_STEPS,
        dt=DT,
        toggle_noise=True,
        no_frames=NO_FRAMES,
        scheme="FE",
        model="Schelling+Voter",
        verbatum=False,
        diagnostic_interval=None,
        reset_projection_diag=True,
        use_fastpath=False,
    )

    runtime = time.perf_counter() - t0
    obs = measure_run(phi_run)

    diag = sim._ensure_work()["projection_diag"]

    diag_summary = {
        "projection_calls": diag.get("n_calls", 0),
        "low_entries": diag.get("n_low_entries", 0),
        "high_entries": diag.get("n_high_entries", 0),
        "simplex_cells": diag.get("n_simplex_cells", 0),
        "max_low_violation": diag.get("max_low_violation", 0.0),
        "reaction_candidates": diag.get("n_schelling_candidates", 0),
        "reaction_events": diag.get("n_schelling_events", 0),
        "reaction_max_candidates": diag.get("max_schelling_candidates", 0),
    }

    return obs, runtime, diag_summary


# ============================================================
# ENSEMBLE
# ============================================================

def run_ensemble():
    results = {mode: [] for mode in MODES}
    runtimes = {mode: [] for mode in MODES}
    diagnostics = {mode: [] for mode in MODES}

    for run_idx in range(N_RUNS):
        seed = BASE_SEED + run_idx

        print(f"\nRun {run_idx + 1}/{N_RUNS}, seed={seed}")

        for mode in MODES:
            print(f"  {mode:24s}", end="", flush=True)

            obs, runtime, diag = run_one(mode, seed)

            results[mode].append(obs)
            runtimes[mode].append(runtime)
            diagnostics[mode].append(diag)

            print(f" {runtime:8.1f} s   low={diag['low_entries']}   simplex={diag['simplex_cells']}")

    for mode in MODES:
        results[mode] = np.asarray(results[mode])
        runtimes[mode] = np.asarray(runtimes[mode])

    return results, runtimes, diagnostics


# ============================================================
# SUMMARY
# ============================================================

def print_final_summary(results, runtimes, diagnostics):
    print("\n" + "=" * 105)
    print("FINAL-FRAME ENSEMBLE COMPARISON")
    print("=" * 105)

    i0 = MODES[0]
    i1 = MODES[1]

    print(f"\n{'observable':22s} {'face_reaction':>18s} {'biased':>18s} {'difference':>15s} {'z':>9s}")

    for j, name in enumerate(OBS_NAMES):
        x = results[i0][:, -1, j]
        y = results[i1][:, -1, j]

        mx = x.mean()
        my = y.mean()
        sx = x.std(ddof=1)
        sy = y.std(ddof=1)

        se = np.sqrt(sx**2 / len(x) + sy**2 / len(y))
        z = (my - mx) / se if se > 0 else np.nan

        print(f"{name:22s} {mx:9.4e} ± {sx:7.2e} {my:9.4e} ± {sy:7.2e} {my-mx:+15.4e} {z:+9.2f}")

    print("\nRuntime")
    for mode in MODES:
        print(f"{mode:24s}: {runtimes[mode].mean():.1f} ± {runtimes[mode].std(ddof=1):.1f} s")

    print("\nProjection diagnostics summed across runs")
    for mode in MODES:
        low = sum(d["low_entries"] for d in diagnostics[mode])
        high = sum(d["high_entries"] for d in diagnostics[mode])
        simplex = sum(d["simplex_cells"] for d in diagnostics[mode])
        max_low = max(d["max_low_violation"] for d in diagnostics[mode])

        print(f"{mode:24s}: low={low:8d}, high={high:8d}, simplex={simplex:8d}, max_low={max_low:.3e}")


# ============================================================
# TIME SERIES PLOTS
# ============================================================

def plot_observable(results, obs_name):
    j = OBS_NAMES.index(obs_name)
    times = np.linspace(0.0, N_STEPS * DT, results[MODES[0]].shape[1])

    fig, ax = plt.subplots()

    for mode in MODES:
        x = results[mode][:, :, j]
        mean = x.mean(axis=0)
        sem = x.std(axis=0, ddof=1) / np.sqrt(N_RUNS)

        ax.plot(times, mean, label=mode)
        ax.fill_between(times, mean - sem, mean + sem, alpha=0.2)

    ax.set_xlabel("time")
    ax.set_ylabel(obs_name)
    ax.legend()
    fig.tight_layout()


def plot_final_distributions(results):
    for obs_name in ("dissimilarity", "var_pol", "var_0"):
        j = OBS_NAMES.index(obs_name)

        fig, ax = plt.subplots()

        data = [results[mode][:, -1, j] for mode in MODES]
        ax.boxplot(data, labels=MODES)

        ax.set_ylabel(obs_name)
        ax.set_title(f"Final {obs_name}")
        fig.tight_layout()


# ============================================================
# SAVE
# ============================================================

def save_results(results, runtimes):
    np.savez_compressed(
        "face_reaction_bias_benchmark.npz",
        face_reaction=results["face_reaction"],
        face_reaction_biased=results["face_reaction_biased"],
        runtime_face_reaction=runtimes["face_reaction"],
        runtime_face_reaction_biased=runtimes["face_reaction_biased"],
        observables=np.array(OBS_NAMES),
        dt=DT,
        nsteps=N_STEPS,
        D=D,
        D_v=D_V,
        beta=BETA,
        kappa=KAPPA,
        Gamma=GAMMA,
        h=H,
        omega=OMEGA,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("FACE REACTION vs BIASED FACE REACTION")
    print("=" * 80)
    print(f"N runs   = {N_RUNS}")
    print(f"N         = {N}")
    print(f"dt        = {DT}")
    print(f"steps     = {N_STEPS}")
    print(f"D         = {D}")
    print(f"D_v       = {D_V}")
    print(f"beta      = {BETA}")
    print(f"kappa     =\n{KAPPA}")
    print(f"Gamma     =\n{GAMMA}")
    print(f"Omega     = {OMEGA}")
    print(f"h         = {H}")

    results, runtimes, diagnostics = run_ensemble()

    print_final_summary(results, runtimes, diagnostics)

    for name in ("dissimilarity", "relative_entropy", "var_pol", "var_occ", "var_0"):
        plot_observable(results, name)

    plot_final_distributions(results)
    save_results(results, runtimes)

    plt.show()


if __name__ == "__main__":
    main()