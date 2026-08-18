import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, irfft
from matplotlib.animation import FuncAnimation, PillowWriter
import fhd
import time
import os
import traceback

import numba as nb

import multiprocessing
import json
import csv
from pathlib import Path

'''
Exploring the parameter space of the (fluctuating) Schelling-Voter model by simulations

Explore for three parameter regions:

- segregating:   kappa = [[0.6, -0.4],[-0.4, 0.6]]
- integrating:  kappa = [[0.6, 1], [1, 0.6]]
- migrating:     kappa = [[1, 1], [-1, 1]]
- well-mixed:    kappa = [[0., 0.], [0., 0.]]

'''
#Helper functions for saving projection diagnostics:

def _to_builtin(x):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _to_builtin(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_builtin(v) for v in x]
    return x


def save_projection_diagnostics(simulator, out_base, metadata=None):
    """
    Save cumulative projection diagnostics.

    Writes:
        <out_base>_projection_diag.json
        <out_base>_projection_history.csv   if history is nonempty
    """
    metadata = {} if metadata is None else dict(metadata)

    work = getattr(simulator, "_work", None)
    if work is None or "projection_diag" not in work:
        print(f"No projection diagnostics found for {out_base}", flush=True)
        return

    diag = work["projection_diag"]

    # Separate cumulative summary from optional per-record history.
    history = diag.get("history", [])
    summary = {k: v for k, v in diag.items() if k != "history"}

    # Useful derived bookkeeping check.
    expected_net = (
        summary.get("mass_added_low", 0.0)
        - summary.get("mass_removed_high", 0.0)
        - summary.get("mass_removed_simplex", 0.0)
        + summary.get("mass_added_transfer_fallback", 0.0)
        + summary.get("mass_roundoff_cleanup", 0.0)
    )

    summary["expected_net_change"] = expected_net
    summary["bookkeeping_error"] = (
        summary.get("net_mass_change_projection", 0.0) - expected_net
    )

    output = {
        "metadata": metadata,
        "projection_summary": summary,
    }

    json_path = Path(f"{out_base}_projection_diag.json")
    tmp_json_path = json_path.with_suffix(json_path.suffix + ".tmp")

    with open(tmp_json_path, "w") as f:
        json.dump(_to_builtin(output), f, indent=2)

    os.replace(tmp_json_path, json_path)

    # Optional sparse history, only if you recorded it.
    if len(history) > 0:
        csv_path = Path(f"{out_base}_projection_history.csv")
        tmp_csv_path = csv_path.with_suffix(csv_path.suffix + ".tmp")

        fieldnames = sorted(set().union(*(h.keys() for h in history)))

        with open(tmp_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in history:
                writer.writerow(_to_builtin(row))

        os.replace(tmp_csv_path, csv_path)

# RUN SWEEP CODE:

N = (2**7, 2**7)
L = (50,50)
Lx, Ly = L
simulator = fhd.fhd_2d(L,N, bc= 'Neumann', fft=False, projection_mode="clip")

regimes = ["segregating", "integrating", "migrating", "well-mixed"]
kappas = [np.array([[0.6, -0.4],[-0.4, 0.6]]), 
          np.array([[0.6, 1], [1, 0.6]]),
          np.array([[1, 1], [-1, 1]]), 
          np.array([[0., 0.], [0., 0.]])
          ]


D = 0.1*np.ones(2) # diffusion coefficient
Gamma = 1*np.eye(2) # Utility nabla^3 term coefficient
D_v = 0.00
noise_v = 1
beta = 10
h = 0.01

param = {'D': D, 'Gamma': Gamma, 'D_v' : D_v, 'beta': beta, 'h': h}

# Parameter values for D - D_v sweep 
Dv_vals = np.linspace(0.01,0.2,20) # D range between 0 and 0.1
num_sims = np.arange(10)


nspecies = simulator.nspecies
N = simulator.N
L = simulator.L


dt = 1e-3
nsteps = 2_000_000
noise = False
frames = 800

no_cores = 128

def run_simulation(param_set):
    try:
        regime, n_run, D_v = param_set
        pid = os.getpid()
        seed = (100*regimes.index(regime) + 10_000*int(n_run) + int(round(1_000_000 * float(D_v))))
        np.random.seed(seed)
        local_simulator = fhd.fhd_2d(L,N, bc= 'Neumann', fft=False, 
                                     schelling_flux="finite_volume",
                                     projection_mode="clip", 
                                     use_numba_projection=False,
                                     )
        local_simulator.set_seed(seed)

        print(f"[pid={pid}] Task started: {regime}, run {n_run}, Dv = {D_v}", flush=True)

        local_param = param.copy()
        local_param["D_v"] = D_v

        if regime == "segregating":
            local_param['kappa'] = kappas[0]
        elif regime == "integrating":
            local_param['kappa'] = kappas[1]
        elif regime == "migrating":
            local_param['kappa'] = kappas[2]
        elif regime == "well-mixed":
            local_param['kappa'] = kappas[3]



        phi = np.zeros((2,) + local_simulator.N)
        phi0 = 0.35
        phi[0] = phi0 + 0.05 * np.random.normal(size=local_simulator.N)
        phi[1] = phi0 + 0.05 * np.random.normal(size=local_simulator.N)

        os.makedirs(f"data/{regime}", exist_ok=True)

    
        st = time.time()
        phi_run = local_simulator.run(
            phi,
            local_param,
            nsteps,
            dt,
            noise,
            no_frames=frames,
            scheme="FE",
            model="Schelling+Voter",
            use_fastpath=True,
        )

        et = time.time()

        out_base = f"data/{regime}/VS_run_{n_run}_det_Dv{D_v:.3f}"

        np.save(f"{out_base}.npy", phi_run)

        save_projection_diagnostics(
            local_simulator,
            out_base,
            metadata={
                "regime": regime,
                "run": int(n_run),
                "D_v": float(D_v),
                "seed": int(seed),
                "dt": float(dt),
                "nsteps": int(nsteps),
                "noise": bool(noise),
                "frames": int(frames),
                "scheme": "FE",
                "bc": "Neumann",
                "schelling_flux": "finite_volume",
                "projection_mode": local_simulator.projection_mode,
                "projection_floor": float(local_simulator.projection_floor),
                "projection_tol": float(getattr(local_simulator, "projection_tol", 0.0)),
                "runtime_seconds": float(et - st),
            },
        )

        print(
            f"[pid={pid}] Finished and saved {regime}, run {n_run}, "
            f"D_v={D_v:.3f}, t={et-st:.6f} s",
            flush=True,
        )

        print(
            f"[pid={pid}] Finished {regime}, run {n_run}, D_v={D_v:.3f}, "
            f"t={et-st:.6f} s",
            flush=True,
        )

        np.save(f"data/{regime}/VS_run_{n_run}_det_Dv{D_v:.3f}.npy", phi_run)
        print(f"[pid={pid}] Saved {regime}, run {n_run}, D_v={D_v:.3f}", flush=True)

            # phi = phi_run[:, -1].copy() Uncomment to start next D_v at last D_v.

    except Exception:
        print(f"ERROR in task {param_set}", flush=True)
        traceback.print_exc()
        raise

parameter_sets = [
    (regime, n_run, Dv) 
    for regime in regimes 
    for n_run in num_sims
    for Dv in Dv_vals
]

def parallel_simulation(parameter_sets):
    print(f"Number of tasks: {len(parameter_sets)}", flush = True)
    print(f"Number of worker processes: {no_cores}", flush = True)
    with multiprocessing.Pool(no_cores) as pool:
        pool.map(run_simulation, parameter_sets)
    
    return 

if __name__ == '__main__':
    parallel_simulation(parameter_sets)


