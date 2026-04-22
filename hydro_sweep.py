import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, irfft
from matplotlib.animation import FuncAnimation, PillowWriter
import fhd
import time
import os

import multiprocessing

'''
Exploring the parameter space of the (fluctuating) Schelling-Voter model by simulations

'''

#sweep 1: deterministic hydro theta = 2/5 [D_v, rho_0]-plane #Redo sweep1
#sweep 2: deterministic hydro D_v = 0.001 [kab, kba]-plane #redo and Run longer! (frames = 200, dt = 0.002, n_steps = 2000000)
#sweep 3: fluctuating hydro (no VOTER noise!) D_v = 0.001 [kab,kba]-plane (frames = 200, dt = 0.005, n_steps = 400000)
#sweep 4: fluctuating hydro D_v = 0.001 [kab,kba]-plane (frames = 200, dt = 0.002, nsteps = 600000) 
#sweep 5: fluctuating Schelling hydro (No Voter) (frames = 200, dt = 0.002, nsteps = 400000)
#sweep 6: deterministic hydro D_v = 0.1 [kab,kba] - plane (frames = 200, dt = 0.002, n_steps = 2000000)
#sweep 7: fluctuating hydro D_V = 0.1 [kab,kba]-plane (frames = 200, dt = 0.002, nsteps = 600000) 

#testsweep 1: 2D fluctuating hydro h = 0, D_v = 0.001, [kab,kba]- plane in 11 steps (frame = 200, dt = 0.002, nsteps = 200000)
#sweep2d1: 2D fluctuating hydro h = 0.1, D_v = 0.001, [kab,kba]- plane in 21 steps (frame = variable, dt = 0.002, nsteps = 500*frames, T=200, K=100)
#sweep2d2: 2D fluctuating hydro h = 0.1, D_v = 0.1, [kab,kba]- plane in 21 steps (frame = variable, dt = 0.002, nsteps = 500*frames, T=200, K=100)
#sweep2d3: 2D fluctuating hydro h = 0.1, D_v = 0.01, [kab,kba]- plane in 21 steps (frame = variable, dt = 0.002, nsteps = 500*frames, T=200, K=100)
#sweep2d4: 2D deterministic hydro D_v = 0.001, [kab,kba]- plane in 21 steps (frame = variable, dt = 0.002, nsteps = 500*frames, T=200, K=100)
#sweep2d5: 2D deterministic hydro D_v = 0.01, [kab,kba]- plane in 21 steps (frame = variable, dt = 0.002, nsteps = 500*frames, T=200, K=100)
#sweep2d6: 2D deterministic hydro D_v = 0.1, [kab,kba]- plane in 21 steps (frame = variable, dt = 0.002, nsteps = 500*frames, T=200, K=100)

#sweep_seg1: 2D deterministic hydro D_v = 0.001, [kab,kba]- plane in 21 steps (frame = variable, dt = 0.002, nsteps = 500*frames, T=200, K=100) from segregated IC!

#sweep_VS: 2D deterministic hydro kab = [[0.4,-0.6],[-0.6,0.4]], [D_v, D_s]-plane (0-1) (frame = variable, dt = 0.002, nsteps = 500*frames, T=200, K=100)
#sweep_VS1: 2D fluctuating hydro kab = [[0.4,-0.6],[-0.6,0.4]], [D_v, D_s]-plane (0-1) (frame = variable, dt = 0.002, nsteps = 500*frames, T=200, K=100)
#sweep_VS2: 2D fluctuating hydro kab = [[0.4,-0.6],[-0.6,0.4]], [D_v, D_s]-plane (0-0.1, 0-1.0)(frame = variable, dt = 0.002, nsteps = 500*frames, T=200, K=100)
#sweep_VS3: 2D deterministic hydro kab = [[0.4,-0.6],[-0.6,0.4]], [D_v, D_s]-plane (0-0.1)(frame = variable, dt = 0.002, nsteps = 500*frames, T=200, K=100)

# RUN SWEEP CODE:

N = (2**6, 2**6)
L = (50,50)
Lx, Ly = L
simulator = fhd.fhd_2d(L,N, bc= 'periodic', fft=False)

T = 2/5
kappa_aa = kappa_bb = 1 - T
kappa_ab = - T 
kappa_ba = - T
# coupling matrix kappa  for pi^{(a)} = sum_b kappa_ab * (V * phi_b)
kappa = np.array([[kappa_aa, kappa_ab],
                  [kappa_ba, kappa_bb]])
D = 0.1*np.ones(2) # diffusion coefficient
Gamma = 1*np.eye(2) # Utility nabla^3 term coefficient
D_v = 0.001
noise_v = 1
beta = 1/D[0]

param = {'D': D, 'kappa': kappa, 'Gamma': Gamma, 'D_v' : D_v, 'beta': beta, 'noise_v': noise_v, 'h': 0.1}

# Parameter values for D - D_v sweep 
p1_vals = np.arange(0, 1.0001, 0.05) # D range between 0 and 1
p2_vals = np.arange(0, 0.1001, 0.005) # D_v range between 0 and 0.1


# For k-ab sweeps
# p1_vals = np.arange(-1.0, 1.0001, 0.1)
# p2_vals = np.arange(-1.0, 1.0001, 0.1)

x = simulator.x
nspecies = simulator.nspecies
N = simulator.N
L = simulator.L


H_indices = np.zeros((len(p1_vals), len(p2_vals)))
D_indices = np.zeros((len(p1_vals), len(p2_vals)))

dt = 0.002
nsteps = 800000
noise = True
frames = 400

x = simulator.x
y = simulator.y
xx, yy = np.meshgrid(x, y, indexing='ij')
nspecies = simulator.nspecies

phi = np.zeros((nspecies,)+ simulator.N)
phi0 = 0.35
pert_amp = 0.0
phi[0] = phi0 + pert_amp * np.cos(2*np.pi*(xx) / Lx * 2)  + pert_amp * np.cos(2*np.pi*(yy) / Lx * 2)  + 0.05 * np.random.normal(size=simulator.N) 
phi[1] = phi0 + pert_amp * np.sin(2*np.pi*(xx) / Lx * 2) + pert_amp * np.sin(2*np.pi*(yy) / Lx * 2) + 0.05 * np.random.normal(size=simulator.N) 

# phi = np.load("data/sweep2d4/run_kdiag06_Dv_0001_kab-1.00_kba-1.00.npy")[:,-1,:,:]

no_cores = 6

def run_simulation(param_set):
    D, Dv = param_set

    param["D"] = D*np.ones(2)
    param["D_v"] = Dv
    # kappa_ab, kappa_ba = param_set
    # param['kappa'] = np.array([[kappa_aa, kappa_ab],
                #   [kappa_ba, kappa_bb]])
    
    # seed = int(time.time())%1000000 + os.getpid() + int(1000*kappa_ab) + int(100*kappa_ba)
    seed = os.getpid() + int(10000*D) + int(100*Dv)
    np.random.seed(seed)
    # print(f"process: {os.getpid()}, random seed = {seed} \n")
    
    phi_start = phi.copy()
    print(f"Starting D = {D:.2f}, D_v = {Dv:.3f} \n")
    # print(f"Starting kappa_ab = {kappa_ab:.2f}, kappa_ba = {kappa_ba:.2f} \n")
    st = time.time()
    result = simulator.run_until_converged(phi_start, param, dt, noise, T=200, K=200, model = "Schelling+Voter", verbatum=False)
    et = time.time()
    print(f"Simulation finished in t = {et-st:.6f}s ")
    np.save(f"data/sweep_VS2/run_T04_D{D:.2f}_Dv{Dv:.3f}", result)
    return 

parameter_sets = [(param_1_value, param_2_value) for param_1_value in p1_vals for param_2_value in p2_vals]

def parallel_simulation(parameter_sets):
    with multiprocessing.Pool(no_cores) as pool:
        pool.map(run_simulation, parameter_sets)
    
    return 

if __name__ == '__main__':
    parallel_simulation(parameter_sets)