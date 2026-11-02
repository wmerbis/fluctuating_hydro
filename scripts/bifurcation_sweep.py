import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, irfft
from matplotlib.animation import FuncAnimation, PillowWriter
import fhd
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

N = (2**7, 2**7)
L = (50,50)
Lx, Ly = L
simulator = fhd.fhd_2d(L,N, bc= 'periodic', fft=False)

regimes = ["segregating", "integrating", "migrating", "well-mixed"]
# regimes = ["migrating", "well-mixed"]
kappas = [np.array([[0.6, -0.4],[-0.4, 0.6]]), np.array([[0.6, 1], [1, 0.6]]),
np.array([[1, 1], [-1, 1]]), np.array([[0.6, 0.4], [0.6, 0.4]])]

# T = 2/5
# kappa_aa = kappa_bb = 1 - T
# kappa_ab = - T 
# kappa_ba = - T
# coupling matrix kappa  for pi^{(a)} = sum_b kappa_ab * (V * phi_b)
# kappa = np.array([[kappa_aa, kappa_ab],
#                   [kappa_ba, kappa_bb]])
D = 0.1*np.ones(2) # diffusion coefficient
Gamma = 1*np.eye(2) # Utility nabla^3 term coefficient
D_v = 0.00
noise_v = 1
beta = 1/D[0]

param = {'D': D, 'kappa': kappas[0], 'Gamma': Gamma, 'D_v' : D_v, 'beta': beta, 'noise_v': noise_v, 'h': 0.1}

# Parameter values for D - D_v sweep 
Dv_vals = np.linspace(0.01,0.2,20) # D range between 0 and 0.1
num_sims = np.arange(10)

x = simulator.x
nspecies = simulator.nspecies
N = simulator.N
L = simulator.L


dt = 0.002
nsteps = 800000
noise = True
frames = 400

x = simulator.x
y = simulator.y
xx, yy = np.meshgrid(x, y, indexing='ij')
nspecies = simulator.nspecies


# phi = np.load("data/sweep2d4/run_kdiag06_Dv_0001_kab-1.00_kba-1.00.npy")[:,-1,:,:]

no_cores = 8

def run_simulation(param_set):
    try:
        regime, n_run = param_set
        pid = os.getpid()
        local_simulator = fhd.fhd_2d(L, N, bc='periodic', fft=False)

        print(f"[pid={pid}] Task started: {regime}, run {n_run}", flush=True)

        local_param = param.copy()

        if regime == "segregating":
            local_param['kappa'] = kappas[0]
        elif regime == "integrating":
            local_param['kappa'] = kappas[1]
        elif regime == "migrating":
            local_param['kappa'] = kappas[2]
        elif regime == "well-mixed":
            local_param['kappa'] = kappas[3]

        seed = pid + int(10*n_run)
        np.random.seed(seed)

        phi = np.zeros((2,) + local_simulator.N)
        phi0 = 0.35
        phi[0] = phi0 + 0.05 * np.random.normal(size=local_simulator.N)
        phi[1] = phi0 + 0.05 * np.random.normal(size=local_simulator.N)

        os.makedirs(f"data/{regime}", exist_ok=True)

        for i, Dv in enumerate(Dv_vals):
            print(f"[pid={pid}] Starting {regime}, run {n_run}, D_v={Dv:.2f}", flush=True)

            local_param["D_v"] = Dv

            st = time.time()
            phi_run = local_simulator.run_until_converged(
                phi,
                local_param,
                dt,
                noise,
                T=200,
                K=100,
                scheme='FE',
                model="Schelling+Voter",
                verbatum=False,
            )
            et = time.time()

            print(
                f"[pid={pid}] Finished {regime}, run {n_run}, D_v={Dv:.2f}, "
                f"t={et-st:.6f} s",
                flush=True,
            )

            np.save(f"data/{regime}/run_{n_run}_fluc_Dv{Dv:.2f}.npy", phi_run)
            print(f"[pid={pid}] Saved {regime}, run {n_run}, D_v={Dv:.2f}", flush=True)

            phi = phi_run[:, -1].copy()

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


