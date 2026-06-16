import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, irfft
from matplotlib.animation import FuncAnimation, PillowWriter
import fhd
from finite_volume_stochastic import *
import time
import os
import traceback

import multiprocessing


'''
Exploring the parameter space of the (fluctuating) Schelling-Voter model by simulations

Explore for three parameter regions:

- segregating:   kappa = [[0.6, -0.4],[-0.4, 0.6]]
- integrsating:  kappa = [[0.6, 1], [1, 0.6]]
- migrating:     kappa = [[1, 1], [-1, 1]]
- well-mixed:    kappa = [[0.6, 0.4], [0.6, 0.4]]

'''

# RUN SWEEP CODE:
nx, ny = 128, 128
lx, ly = 50, 50
dt = 5e-4
nsteps = 2_000_000
frames = 2000
store_every = nsteps//frames


# regimes = ["segregating", "integrating", "migrating", "well-mixed"]
regimes = ["well-mixed"]
kappas = [np.array([[0.6, -0.4],[-0.4, 0.6]]), np.array([[0.6, 1], [1, 0.6]]),
np.array([[1, 1], [-1, 1]]), np.array([[0.6, 0.4], [0.6, 0.4]])]


# Parameter values for D - D_v sweep 
Dv_vals = np.linspace(0.01,0.2,20) # D range between 0 and 0.1
num_sims = np.arange(10)

no_cores = 128

def run_simulation(param_set):
    try:
        regime, n_run = param_set
        pid = os.getpid()
        
        print(f"[pid={pid}] Task started: {regime}, run {n_run}", flush=True)


        if regime == "segregating":
            kappa = kappas[0]
        elif regime == "integrating":
            kappa = kappas[1]
        elif regime == "migrating":
            kappa = kappas[2]
        elif regime == "well-mixed":
            kappa = kappas[3]


        os.makedirs(f"data/{regime}", exist_ok=True)
        rho_A0, rho_B0 = make_random_initial_condition(nx, ny, seed=1)

        for i, Dv in enumerate(Dv_vals):
            print(f"[pid={pid}] Starting {regime}, run {n_run}, D_v={Dv:.2f}", flush=True)
            
            seed = pid + int(10*n_run) + int(1000*Dv)

            local_params = ModelParameters(
                D_A=0.1,
                D_B=0.1,
                D_v=Dv,
                beta=10.0,
                h_noise=min(lx / nx, ly / ny),
                kappa=kappa,
                gamma=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
            )

            solver = SchellingVoterFVSolver(
                nx=nx,
                ny=ny,
                lx=lx,
                ly=ly,
                params=local_params,
                bc="periodic",
                semi_implicit_stiff=True,
                stochastic=True,
                rng=np.random.default_rng(seed),
                limiter_mode= 'clip'
            )

            st = time.time()
            solver.set_state(rho_A0, rho_B0)
            result = solver.run(dt=dt, nsteps=nsteps, snapshot_every=store_every, add_noise=True, verbatum = False)
            et = time.time()

            print(
                f"[pid={pid}] Finished {regime}, run {n_run}, D_v={Dv:.2f}, "
                f"t={et-st:.6f} s",
                flush=True,
            )

            frames = len(result['snapshots'])
            phi_run = np.array([[result['snapshots'][t][1],result['snapshots'][t][2]] for t in range(frames)])
            phi_run = phi_run.transpose((1,0,2,3))
            np.save(f"data/{regime}/FV_run_{n_run}_fluc_Dv{Dv:.2f}.npy", phi_run)
            print(f"[pid={pid}] Saved {regime}, run {n_run}, D_v={Dv:.2f}", flush=True)

            rho_A0, rho_B0 = result["rho_A"] , result["rho_B"]


    except Exception:
        print(f"ERROR in task {param_set}", flush=True)
        traceback.print_exc()
        raise

parameter_sets = [(param_1_value, param_2_value) for param_1_value in regimes for param_2_value in num_sims]

def parallel_simulation(parameter_sets):
    with multiprocessing.Pool(no_cores) as pool:
        pool.map(run_simulation, parameter_sets)
    
    return 

if __name__ == '__main__':
    parallel_simulation(parameter_sets)


