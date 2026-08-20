import numpy as np
import glob
import matplotlib.pyplot as plt
from pathlib import Path
import re
from numpy.fft import fft2, ifft2, fftshift
# ----------------------------
# set your scan values here
# ----------------------------
n_Da  = 11
n_Dv  = 11
n_rep = 10
M     = 200
where_to_start = 5
# Replace these with the actual values you used in the scan 
Dv_vals = np.linspace(0.01, 0.1, 11)   # 11×11 grid
Da_vals = np.linspace(0.01, 1, 11)

# ----------------------------
# allocate heatmap arrays
# ----------------------------

ka = np.full((n_Da, n_Dv), np.nan)
kp = np.full((n_Da, n_Dv), np.nan)
k0_map = np.full((n_Da, n_Dv), np.nan)
slopeA_map = np.full_like(k0_map, np.nan)
slopeB_map = np.full_like(k0_map, np.nan)
slopeAB_map = np.full_like(k0_map, np.nan)

# ----------------------------
# fixed parameters for title
# ----------------------------
vacant = 0.3
kappa_aa = 0.6
kappa_ab = -0.4
Gamm = 1
Gamma_aa = Gamm
Gamma_bb = Gamm
Gamma_ab = 0
Gamma_ba = 0
beta = 10
Nx = 128
Ny = 128
N = Nx * Ny

# ----------------------------------------------------------------
# define physical k axes (if domain lengths Lx,Ly, and spacing dx and dy)
Lx=50
dx = Lx/Nx # in general, if we know the physical length Lx, and the number of houses along one axis, Nx then dx= Lx/Nx, the same for dy=Ly/Ny 
dy = Lx/Nx
kx = 2*np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2*np.pi * np.fft.fftfreq(Nx, d=dy)
kxg, kyg = np.meshgrid(kx, ky, indexing='ij')
kgrid = np.sqrt(kxg**2 + kyg**2)
k_flat = kgrid.ravel()

# -------------------- radial binning procedure -----------------------------
kmax = k_flat.max()
nbins = min(60, Nx//2)   # choose reasonable number bins
kbins = np.linspace(0.0, kmax, nbins+1)
kcent = 0.5*(kbins[:-1] + kbins[1:])
bin_index = np.searchsorted(kbins, k_flat) - 1
valid_bin = (bin_index >= 0) & (bin_index < nbins)
counts = np.bincount(bin_index[valid_bin], minlength=nbins)

mask = counts > 0

def radial_average_2d(S2d):
    Sflat = S2d.ravel().real
    Srad = np.bincount(
        bin_index[valid_bin],
        weights=Sflat[valid_bin],
        minlength=nbins
    ).astype(float)
    Srad[mask] /= counts[mask]
    return Srad
# ----------------------------
# read files
# ----------------------------
folder = Path("results_Dv_scan")
files = sorted(folder.glob("scan_Da_*_Dv_*.npz"))

pattern = re.compile(r"scan_Da_([0-9.]+)_Dv_([0-9.]+)\.npz")

for f in files:
    m = pattern.fullmatch(f.name)
    if m is None:
        print("Skipping:", f.name)
        continue

    Da = float(m.group(1))
    Dv = float(m.group(2))

    i = np.argmin(np.abs(Da_vals - Da))   # Da index
    j = np.argmin(np.abs(Dv_vals - Dv))   # Dv index
    data = np.load(f)
    k0_map[i, j] = data["k0"]
    slopeA_map[i, j] = data["slopeA"]
    slopeB_map[i, j] = data["slopeB"]
    slopeAB_map[i, j] = data["slopeAB"]
    snapshots_all = data["snapshots"] 

    n_take = data["n_take"] if "n_take" in data else np.full(n_rep, M)

    # store time-dependent radial spectra: shape (rep, time, k-bin)
    S_AA_rep = np.full((n_rep, M, nbins), np.nan)
    S_BB_rep = np.full((n_rep, M, nbins), np.nan)
    S_AB_rep = np.full((n_rep, M, nbins), np.nan)
    S_00_rep = np.full((n_rep, M, nbins), np.nan)

    # optional: store characteristic wave number vs time
    kmean_rep = np.full((n_rep, M), np.nan)
    kpeak_rep = np.full((n_rep, M), np.nan)

    # k grid excluding zero mode
    k_vals = kcent[mask]
    pos = k_vals > 0
    k_use = k_vals[pos]

    for rep in range(n_rep):
        take = min(int(n_take[rep]), M)
        snapshots = snapshots_all[rep, :take]

        for t_idx, snap in enumerate(snapshots):
            rhoA = (snap == 1).astype(float)
            rhoB = (snap == -1).astype(float)
            rho0 = (snap == 0).astype(float)

            a0 = rhoA.mean()
            b0 = rhoB.mean()

            dA = rhoA - a0
            dB = rhoB - b0
            d0 = rho0 - rho0.mean()

            dA_k = fft2(dA)
            dB_k = fft2(dB)
            d0_k = fft2(d0)

            S_AA_2d = (dA_k * np.conj(dA_k)).real / N
            S_BB_2d = (dB_k * np.conj(dB_k)).real / N
            S_00_2d = (d0_k * np.conj(d0_k)).real / N
            S_AB_2d = np.real(dA_k * np.conj(dB_k)) / N

            # radial average at THIS time point
            S_AA_t = radial_average_2d(S_AA_2d)
            S_BB_t = radial_average_2d(S_BB_2d)
            S_00_t = radial_average_2d(S_00_2d)
            S_AB_t = radial_average_2d(S_AB_2d)

            S_AA_rep[rep, t_idx, :] = S_AA_t
            S_BB_rep[rep, t_idx, :] = S_BB_t
            S_00_rep[rep, t_idx, :] = S_00_t
            S_AB_rep[rep, t_idx, :] = S_AB_t

            # characteristic k at this time
            Ssym_t = S_AA_t[mask][pos] + S_BB_t[mask][pos]
            if np.any(np.isfinite(Ssym_t)) and np.nansum(Ssym_t) > 0:
                kmean_rep[rep, t_idx] = np.sum(k_use * Ssym_t) / np.sum(Ssym_t)
                kpeak_rep[rep, t_idx] = k_use[np.argmax(Ssym_t)]

    S_AA_mean = np.nanmean(S_AA_rep, axis=0)   # shape (M, nbins)
    S_BB_mean = np.nanmean(S_BB_rep, axis=0)
    S_AB_mean = np.nanmean(S_AB_rep, axis=0)
    S_00_mean = np.nanmean(S_00_rep, axis=0)

    kmean_mean = np.nanmean(kmean_rep, axis=0)
    kmean_sem  = np.nanstd(kmean_rep, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(kmean_rep), axis=0))

    kpeak_mean = np.nanmean(kpeak_rep, axis=0)
    kpeak_sem  = np.nanstd(kpeak_rep, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(kpeak_rep), axis=0))

    ka[i, j] = kmean_mean[-1]
    kp[i, j] = kpeak_mean[-1]
    # t_plot = np.arange(1, M + 1)   # snapshot index; use real saved times if available
    # valid_t = ~np.isnan(kmean_mean)

    # plt.figure()
    # plt.loglog(t_plot[valid_t], kmean_mean[valid_t], label="mean over reps")
    # plt.fill_between(
    #     t_plot[valid_t],
    #     kmean_mean[valid_t] - kmean_sem[valid_t],
    #     kmean_mean[valid_t] + kmean_sem[valid_t],
    #     alpha=0.2
    # )

    # reference slopes
    # tref = t_plot[valid_t]
    # if len(tref) > 0:
    #     c1 = kmean_mean[valid_t][0] * tref[0]**(1/3)
    #     c2 = kmean_mean[valid_t][0] * tref[0]**(1/4)
    #     plt.loglog(tref, c1 * tref**(-1/3), label=r'$\sim t^{-1/3}$')
    #     plt.loglog(tref, c2 * tref**(-1/4), label=r'$\sim t^{-1/4}$')

    # plt.xlabel("t")
    # plt.ylabel(r'$\langle k \rangle$')
    # plt.legend()
    # plt.title(fr"$ Da = {Da_vals[i]},\, Dv = {Dv_vals[j]} $")
    # plt.tight_layout()
    # #plt.show()
    # plt.savefig(f"Da{Da_vals[i]}_Dv{Dv_vals[j]}.png", dpi=300)  

# pattern = re.compile(r"scan_Da_(\d+)_Dv_(\d+)\.npz")

# for f in files:
#     m = pattern.fullmatch(f.name)
#     if m is None:
#         continue

#     i = int(m.group(1))   # Da index
#     j = int(m.group(2))   # Dv index

# pattern = re.compile(r"scan_Da_([0-9.]+)_Dv_([0-9.]+)\.npz")

# for f in files:
#     m = pattern.fullmatch(f.name)
#     if m is None:
#         print("Skipping:", f.name)
#         continue

#     Da = float(m.group(1))
#     Dv = float(m.group(2))

#     i = np.argmin(np.abs(Da_vals - Da))   # Da index
#     j = np.argmin(np.abs(Dv_vals - Dv))   # Dv index
#     data = np.load(f)

#     k0_map[i, j] = data["k0"]
#     slopeA_map[i, j] = data["slopeA"]
#     slopeB_map[i, j] = data["slopeB"]
#     slopeAB_map[i, j] = data["slopeAB"]
#     # --- NEW PART ---
#     if "snapshots" in data:
#         snaps = data["snapshots"]  # (n_rep, M, Ny, Nx)
#         n_rep, M, Ny, Nx = snaps.shape 

#         for r in range(1):
#             final = snaps[r, -1]
#             # 1) final snapshot
#             plt.title(fr"$ Da = {Da_vals[i]},\, Dv = {Dv_vals[j]} $")
#             plt.imshow(final, cmap='bwr', vmin=-1, vmax=1, origin='lower')
#             #plt.show()
#             plt.savefig(f"Da{Da_vals[i]}_Dv{Dv_vals[j]}.png", dpi=300)  

# # ----------------------------
# plotting
# ----------------------------
def plot_heatmap(Z, title):
    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[Da_vals[0], Da_vals[-1], Dv_vals[0], Dv_vals[-1]],
        cmap='rainbow',
        vmin=0
    )
    plt.xlabel(r"$D_a$")
    plt.ylabel(r"$D_v$")
    plt.colorbar(im, label=title)
    plt.title(
        rf"$\beta={beta}, \Gamma={Gamm}, \kappa_{{aa}}=\kappa_{{bb}}={kappa_aa},\ \kappa_{{ab}}=\kappa_{{ba}}={kappa_ab},\ \rho={vacant}$"
    )
    plt.tight_layout()
    plt.show()
plot_heatmap(ka.T,      r'$\langle k \rangle$')
plot_heatmap(kp.T,  r'$k_c$')
plot_heatmap(k0_map.T,      r'$k_c $')
#plot_heatmap(slopeA_map.T,  r'slope $G_{AA}$')
#plot_heatmap(slopeB_map.T,  r'slope $G_{BB}$')
#plot_heatmap(slopeAB_map.T, r'slope $G_{AB}$') 