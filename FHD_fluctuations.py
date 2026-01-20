"""
2D implementation of Periodic  domain
Author: Tuan Pham 
"""
import numpy as np
from numpy.fft import ifft2, fftshift
import matplotlib.pyplot as plt
from scipy.linalg import expm, inv
import time
kappa_aa = 1 
kappa_bb = 1
kappa_ab =-1
kappa_ba =-1
Gamma_aa = 1
Gamma_bb = 1
Gamma_ab = 0
Gamma_ba = 0
A_fixedpoint = 0.35
B_fixedpoint = 0.35
D_A = 0.1
D_B = 0.1
D_v = 0
Nx = 256
Ny = 256
Lx = 256
Ly = 256
h = Lx/Nx
params = {
    'D_A': D_A, 'D_B': D_B, 'D_v': D_v,
    'beta': 1.0/D_A, 'h': h,
    'kappa': [[kappa_aa, kappa_ab],[kappa_ba, kappa_bb]],
    'Gamma': [[Gamma_aa, Gamma_ab],[Gamma_ba, Gamma_bb]],
    'a0': A_fixedpoint, 'b0': B_fixedpoint
}
grid = {'Nx': Nx, 'Ny': Ny, 'Lx': Lx, 'Ly': Ly}
times = np.linspace(0.0, 5.0, 51)
# ---------- Functions (linearized model) ----------
def build_L_matrices(params):
    Da, Db, Dv = params['D_A'], params['D_B'], params['D_v']
    beta = params['beta']
    a0, b0 = params['a0'], params['b0']
    s0 = 1.0 - a0 - b0
    kap = np.array(params['kappa'])
    Gam = np.array(params['Gamma'])
    A0 = np.zeros((2,2)); A1 = np.zeros((2,2))
    pref = lambda D, rho: D * beta * s0 * rho

    A0[0,0] = Da*(s0 + a0) + Dv*b0
    A0[1,1] = Db*(s0 + b0) + Dv*a0
    A0[0,1] = Da*a0 - Dv*a0
    A0[1,0] = Db*b0 - Dv*b0

    A0[0,0] -= pref(Da, a0)*kap[0,0]
    A0[0,1] -= pref(Da, a0)*kap[0,1]
    A0[1,0] -= pref(Db, b0)*kap[1,0]
    A0[1,1] -= pref(Db, b0)*kap[1,1]

    A1[0,0] = -pref(Da, a0) * Gam[0,0]
    A1[0,1] = -pref(Da, a0) * Gam[0,1]
    A1[1,0] = -pref(Db, b0) * Gam[1,0]
    A1[1,1] = -pref(Db, b0) * Gam[1,1]

    return A0, A1

def build_Qk(k2, params, case='A'):
    Da, Db, Dv = params['D_A'], params['D_B'], params['D_v']
    h = params['h']
    a0, b0 = params['a0'], params['b0']
    s0 = 1.0 - a0 - b0
    QA = 2.0 * Da * (h**2) * s0 * a0
    QB = 2.0 * Db * (h**2) * s0 * b0
    Qv = 4.0 * Dv * a0 * b0
    Ddiag = np.array([[QA, 0.0],[0.0, QB]])
    exch = Qv * np.array([[1.0, -1.0],[-1.0, 1.0]])
    Qk = k2 * Ddiag + exch

    return Qk

def compute_W(L, Q, dt):
    I2 = np.eye(2)
    M = np.kron(L, I2) + np.kron(I2, L)
    E = expm(-M * dt)
    try:
        Minv = inv(M)
    except np.linalg.LinAlgError:
        Minv = inv(M + 1e-12 * np.eye(M.shape[0])) #1e-12  to improve robustness:
    vecQ = Q.reshape(4, order='F')
    vecW = (np.eye(4) - E).dot(Minv).dot(vecQ)
    W = vecW.reshape((2,2), order='F')
    W = 0.5 * (W + W.T)
    return W

def evolve_covariances_fast(grid, params, times, case='A', C0k=None):
    Nx, Ny = grid['Nx'], grid['Ny']
    Lx, Ly = grid['Lx'], grid['Ly']
    dx, dy = Lx/Nx, Ly/Ny
    A0, A1 = build_L_matrices(params)
    kx = 2*np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2*np.pi * np.fft.fftfreq(Ny, d=dy)
    kxg, kyg = np.meshgrid(kx, ky, indexing='ij')
    k2_grid = kxg**2 + kyg**2
    k2_vals, inv_idx = np.unique(k2_grid, return_inverse=True)

    if C0k is None:
        Ck = np.zeros((Nx*Ny,2,2), dtype=float)
    else:
        Ck = C0k.reshape((Nx*Ny,2,2)).copy()

    nt = len(times)
    S_AA = np.zeros((Nx,Ny,nt))
    S_BB = np.zeros((Nx,Ny,nt))
    S_AB = np.zeros((Nx,Ny,nt))

    for ti in range(nt):
        # record equal-time covariances
        for idx in range(Nx*Ny):
            i = idx // Ny; j = idx % Ny
            S_AA[i,j,ti] = Ck[idx,0,0]
            S_BB[i,j,ti] = Ck[idx,1,1]
            S_AB[i,j,ti] = Ck[idx,0,1]
        if ti == nt-1:
            break
        dt = times[ti+1] - times[ti]
        # compute Prop and W once per unique k^2
        for u, k2 in enumerate(k2_vals):
            Lk = (k2 * A0) - (k2**2 * A1)
            Qk = build_Qk(k2, params, case=case)
            Prop = expm(-Lk * dt)
            W = compute_W(Lk, Qk, dt)
            indices = np.where(inv_idx == u)[0]
            for idx in indices:
                Ck[idx] = Prop.dot(Ck[idx]).dot(Prop.T) + W
    return {'S_AA': S_AA, 'S_BB': S_BB, 'S_AB': S_AB, 'kx': kx, 'ky': ky}

t0 = time.time()
outA = evolve_covariances_fast(grid, params, times, case='A')
t1 = time.time()
print(f"Evolved covariances for {len(times)} times on {grid['Nx']}x{grid['Ny']} grid in {t1-t0:.2f}s")
# Final equal-time spectrum for species A
S_AA_final = outA['S_AA'][:,:, -1]
S_BB_final = outA['S_BB'][:,:, -1]
S_AB_final = outA['S_AB'][:,:, -1]
# Radially average the 2D spectrum for plotting
Nx, Ny = grid['Nx'], grid['Ny']
dx = grid['Lx']/Nx
kx = 2*np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2*np.pi * np.fft.fftfreq(Ny, d=dx)
kxg, kyg = np.meshgrid(kx, ky, indexing='ij')
kgrid = np.sqrt(kxg**2 + kyg**2)
k_flat = kgrid.ravel()
SA_flat = S_AA_final.ravel()
SB_flat = S_BB_final.ravel()
SAB_flat = S_AB_final.ravel()

kmax = np.max(kgrid)
nbins = 60
kbins = np.linspace(0, kmax, nbins+1)
kcent = 0.5*(kbins[:-1]+kbins[1:])
SradA = np.zeros(nbins)
SradB = np.zeros(nbins)
SradAB = np.zeros(nbins)
counts = np.zeros(nbins)
for i in range(len(k_flat)):
    k = k_flat[i]
    binidx = np.searchsorted(kbins, k) - 1
    if 0 <= binidx < nbins:
        SradA[binidx] += SA_flat[i]
        SradB[binidx] += SB_flat[i]
        SradAB[binidx] += SAB_flat[i]
        counts[binidx] += 1
valid = counts>0
SradA[valid] /= counts[valid]
SradB[valid] /= counts[valid]
SradAB[valid] /= counts[valid]

# -------------------- Fit Srad to C / k^2 and plot --------------------
# kcent, SradA, SradAB, valid are from your code above.

# choose data points to fit: exclude k=0 and any bins with zero count
k_vals = kcent[valid]
Svals_A = SradA[valid]
Svals_AB = SradAB[valid]

# exclude tiny k (first bin) to avoid singularity / dominated-by-k0
# pick indices where k>0 (should be all valid except possibly first)
pos = k_vals > 0
k_fit = k_vals[pos]
SA_fit = Svals_A[pos]
SAB_fit = Svals_AB[pos]

# design variable: x = 1/k^2
X = 1.0 / (k_fit**2)

# least-squares amplitude fit for model y = C * (1/k^2)
# C = sum(y * X) / sum(X^2)
def fit_C(y, X):
    num = np.sum(y * X)
    den = np.sum(X * X)
    if den == 0:
        return 0.0
    return num / den

C_A = fit_C(SA_fit, X)
C_AB = fit_C(SAB_fit, X)

print("Fitted amplitudes: C_A =", C_A, ", C_AB =", C_AB)

# Build fitted curves on same k grid for plotting
kplot = np.linspace(np.min(k_fit), np.max(k_fit), 200)
fitA = C_A / (kplot**2)
fitAB = C_AB / (kplot**2)

# --- Log–log slope extraction ---
def loglog_slope(k, S):
    mask = (k > 0) & (S > 0)
    k = k[mask]
    S = S[mask]
    if len(k) < 2:
        return np.nan, np.nan
    logk = np.log(k)
    logS = np.log(S)
    A = np.vstack([logk, np.ones(len(k))]).T
    sol, *_ = np.linalg.lstsq(A, logS, rcond=None)
    slope, intercept = sol
    return -slope, intercept

slope_A, _ = loglog_slope(k_fit, SA_fit)
slope_AB, _ = loglog_slope(k_fit, np.abs(SAB_fit))
print(f"log–log slope AA = {slope_A:.3f}")
print(f"log–log slope AB = {slope_AB:.3f}")

fitAnew = C_A / (kplot**slope_A)
fitABnew = -C_AB / (kplot**slope_AB)

# Plot data + fits (linear scale)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(k_vals, Svals_A, 'o-', label='S_AA (radial binned)', markersize=5)
plt.plot(k_vals, Svals_AB, 's-', label='S_AB (radial binned)', markersize=5)
#plt.plot(kplot, fitA, '--', label=f'fit AA: {C_A:.3e}/k^2')
#plt.plot(kplot, fitAB, '--', label=f'fit AB: {C_AB:.3e}/k^2')
plt.xlabel('k')
plt.ylabel('qual-time correlations')
plt.legend()
plt.title(fr"$\rho_A^{(0)} = \rho_B^{(0)} ={A_fixedpoint},\, \Gamma = {Gamma_aa},\, D = {D_A},\, \kappa aa = {kappa_aa},\, \kappa ab = {kappa_ab},\, \kappa ba = {kappa_ba}, \, Dv={D_v} $")
plt.grid(True)
# Optional: log-log view to check slope ~ -2
plt.subplot(1,2,2)
plt.loglog(k_fit, SA_fit, 'o', label=f'G_AA with 1/k^{slope_A}')
#plt.loglog(kplot, fitAnew, '--', label=f'fit AA: {C_A:.3e}/k^{slope_A}')
plt.loglog(k_fit,  np.abs(SAB_fit), 's', label=f'G_AB with 1/k^{slope_AB}')
#plt.loglog(kplot, fitABnew, '-', label=f'fit AB: {C_AB:.3e}/k^{slope_AB}')
plt.xlabel('k')
plt.ylabel('G(k)')
plt.legend()
plt.title('Log-log: Check slope')
plt.grid(True, which='both', ls=':')
plt.tight_layout()
plt.show()

#C_AA_real = np.real(ifft2(S_AA_final))
#C_BB_real = np.real(ifft2(S_BB_final))
#C_AB_real = np.real(ifft2(S_AB_final))
# plt.subplot(1,3,2)
# plt.title('C_AA(x) at final time')
# plt.imshow(fftshift(C_AA_real), origin='lower', extent=[0,grid['Lx'],0,grid['Ly']]); plt.colorbar()
# plt.subplot(1,3,3)
# plt.title('C_AB(x) at final time')
# plt.imshow(fftshift(C_AB_real), origin='lower', extent=[0,grid['Lx'],0,grid['Ly']]); plt.colorbar()
