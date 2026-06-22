import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, irfft
from matplotlib.animation import FuncAnimation, PillowWriter
import fhd
import importlib
import time


#LOAD SWEEP CODE:
averaging_bp = -200
N = (2**7, 2**7)
L = (50,50)
Lx, Ly = L

nu = np.zeros((2,2,2))
# coupling matrix kappa  for pi^{(a)} = sum_b kappa_ab * (V * phi_b)
D = 0.1*np.ones(2) # Schelling diffusion coefficient
Gamma = np.eye(2) # Utility nabla^3 term coefficient
D_v = 0.01 #Voter model diffusion term
beta = 10

param = {'D': D, 'Gamma': Gamma, 'nu': nu, 'D_v': D_v, 'beta': beta, 'noise_v' : 1, 'h': 50/128}


num_bins = 60

Dv_space = np.linspace(0.01,0.2,20)

regimes = ["segregating", "integrating", "migrating", "well-mixed"]
kappas = [np.array([[0.6, -0.4],[-0.4, 0.6]]), np.array([[0.6, 1], [1, 0.6]]),
np.array([[1, 1], [-1, 1]]), np.array([[0.6, 0.4], [0.6, 0.4]])]


dt = 1e-3
nsteps = 2_000_000
frames = 800
store_every = nsteps//frames
noise = True


H_indices = np.zeros((len(Dv_space),len(regimes)))
D_indices = np.zeros((len(Dv_space),len(regimes)))
Dkl_indices = np.zeros((len(Dv_space),len(regimes)))
Dkl_variance = np.zeros((len(Dv_space),len(regimes)))
kmeans = np.zeros((len(Dv_space),len(regimes)))
kpeaks = np.zeros((len(Dv_space),len(regimes)))
PS_matrix = np.zeros((len(Dv_space), len(regimes), 3, num_bins-1))
GAB_matrix = np.zeros((len(Dv_space), len(regimes), num_bins-1))

def kmean(power_spectrum, k_bins):
    Z = np.sum(power_spectrum)
    kmean = np.sum(k_bins*power_spectrum)
    return kmean/Z

for r, regime in enumerate(regimes):
    print(f"Starting {regime} regime")
    for i, Dv in enumerate(Dv_space):
        phi_runs = []
        for n_run in range(10):
            try:    
                phi_run = np.load(f"data/{regime}/FV_run_{n_run}_fluc_Dv{Dv:.2f}.npy")
                phi_runs.append(phi_run)
            except:
                print(f"Failed to load {regime} sweep run {n_run} for Dv = {Dv:.2f}")
                # phi_run = np.zeros((2,-averaging_bp+1)+N)
        phi_runs = np.array([phi_run[:,averaging_bp:] for phi_run in phi_runs])

        Dkl_list = [fhd.mean_relative_entropy(phi_runs[n_run,:,-t]) for t in range(-averaging_bp) for n_run in range(phi_runs.shape[0])]
        H_indices[i,r] = np.mean([fhd.entropy_index(phi_runs[n_run,:,-t]) for t in range(-averaging_bp) for n_run in range(phi_runs.shape[0])])
        D_indices[i,r] = np.mean([fhd.dissimilarity(phi_runs[n_run,:,-t]) for t in range(-averaging_bp) for n_run in range(phi_runs.shape[0])])
        Dkl_indices[i,r] = np.mean(Dkl_list)
        Dkl_variance[i,r] = np.var(Dkl_list)

        phi_runs = phi_runs.transpose((1,2,0,3,4))
        phi_runs = phi_runs.reshape((2, phi_runs.shape[1]*phi_runs.shape[2],)+N)
        k_bins, power_spectra, G_AB = fhd.power_spectrum(phi_runs, (Lx, Ly), num_bins=60, averaged = True,  centered = True)

        PS_matrix[i,r] = power_spectra
        GAB_matrix[i,r] = G_AB
        kmeans[i,r] = kmean(power_spectra[0]+power_spectra[1],k_bins)
        kpeaks[i,r] = k_bins[power_spectra[0].argmax()]


# Plot dissimilarity indices
# Finite-volume run
plt.semilogx(Dv_space/D[0], Dkl_indices[:,0], "x:k", label = "Segregating regime")
plt.semilogx(Dv_space/D[0], Dkl_indices[:,1], "x:r", label = "Integrated regime")
plt.semilogx(Dv_space/D[0], Dkl_indices[:,2], "x:", color= 'orange', label = "Migrating regime")
plt.semilogx(Dv_space/D[0], Dkl_indices[:,3], "x:g", label = "Well-mixed regime")
# plt.semilogx(D[0]/Dv_space, Dkl_indices_mig[:], "x:", color='orange', label = "Migrating regime")
plt.xlabel(r"$D_v/D^a$", size=18)
plt.ylabel(r"$D_{\rm KL}(p_{\rm loc}, p_{\rm glob})$", size=18)
plt.title(r"Entropic dissimilarity index", size=18)
plt.ylim([0.05,0.16])
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend(fontsize=12)
plt.savefig("images/dissimilarity_FVrun.pdf")
plt.close()

# Plot configurations from FV runs
run = 0
Dv = 0.04

regimes = ["segregating", "integrating", "migrating"]

fig, ax = plt.subplots(
    2, 3,
    figsize=(12, 7),
    constrained_layout=True
)

ims_top = []
ims_bottom = []

for i, regime in enumerate(regimes):
    phi_run = np.load(f"data/{regime}/FV_run_{run}_fluc_Dv{Dv:.2f}.npy")

    phi_diff = phi_run[0] - phi_run[1]
    phi_0 = 1 - phi_run.sum(axis=0)

    im1 = ax[0, i].imshow(
        phi_diff[-1],
        cmap="RdBu",
        aspect="equal",
        origin="lower",
        extent=[-Lx/2, Lx/2, -Ly/2, Ly/2],
        vmin=-1,
        vmax=1
    )

    im2 = ax[1, i].imshow(
        phi_0[-1],
        cmap="Greens",
        aspect="equal",
        origin="lower",
        extent=[-Lx/2, Lx/2, -Ly/2, Ly/2],
        vmin=0,
        vmax=1
    )

    ax[0, i].axis("off")
    ax[1, i].axis("off")

    ax[0, i].set_title(regime, fontsize=20)

    ims_top.append(im1)
    ims_bottom.append(im2)

# One colorbar per row, aligned with all axes in that row
cbar1 = fig.colorbar(
    ims_top[-1],
    ax=ax[0, :],
    location="right",
    fraction=0.035,
    pad=0.02
)
cbar1.set_label(r"$\rho_A - \rho_B$", size=20)

cbar2 = fig.colorbar(
    ims_bottom[-1],
    ax=ax[1, :],
    location="right",
    fraction=0.035,
    pad=0.02
)
cbar2.set_label(r"$\rho_0$", size=20)
plt.savefig(f"images/FV_schelling_phases_noise_Dv{Dv:.2f}.pdf")
plt.close()


# Plot power spectra
from structure_factor import *
import numpy as np
import matplotlib.pyplot as plt

regimes = ["segregating", "integrating", "migrating", "well-mixed"]

kappas = [
    np.array([[0.6, -0.4], [-0.4, 0.6]]),
    np.array([[0.6,  1.0], [ 1.0, 0.6]]),
    np.array([[1.0,  1.0], [-1.0, 1.0]]),
    np.array([[0.6,  0.4], [ 0.6, 0.4]])
]

Dv_indices = [0, 19]

# Large-k cutoff
k_max_cutoff = 4

# Number of theory points to omit from display
theory_skip = 0

# ---------- styling for paper ----------
title_fs  = 16
label_fs  = 15
tick_fs   = 13
legend_fs = 12
text_fs   = 12

lw_th     = 2.0
lw_ref    = 2.0
ms        = 5.5

plt.rcParams.update({
    "font.size": tick_fs,
    "axes.labelsize": label_fs,
    "axes.titlesize": title_fs,
    "xtick.labelsize": tick_fs,
    "ytick.labelsize": tick_fs,
    "legend.fontsize": legend_fs,
})

# ---------- helper functions ----------
def fit_amplitude_logspace(theory, data):
    """
    Fit a multiplicative amplitude amp in log-space:
        log(data) ~ log(amp) + log(theory)
    """
    valid = (
        np.isfinite(theory) & np.isfinite(data) &
        (theory > 0) & (data > 0)
    )

    if not np.any(valid):
        return 1.0

    log_amp = np.mean(np.log(data[valid]) - np.log(theory[valid]))
    return np.exp(log_amp)


def get_k_mask(k_bins, k_max_cutoff=None):
    mask = (k_bins > 0)
    if k_max_cutoff is not None:
        mask &= (k_bins <= k_max_cutoff)
    return mask


def get_dynamic_fit_mask(k_bins, k_mask, power_spectra, G_AB, n_after_peak=2):
    """
    Start the fit 2 points after the maximum of the data envelope
    from S_AA, S_BB, and |S_AB|.
    """
    envelope = np.nanmax(
        np.vstack([
            np.where(k_mask, power_spectra[0], np.nan),
            np.where(k_mask, power_spectra[1], np.nan),
            np.where(k_mask, np.abs(G_AB), np.nan)
        ]),
        axis=0
    )

    if np.all(np.isnan(envelope)):
        idx_start = 1
    else:
        idx_start = np.nanargmax(envelope) + n_after_peak

    idx = np.arange(len(k_bins))
    fit_mask = k_mask & (idx >= idx_start)

    if np.sum(fit_mask) < 3:
        fit_mask = k_mask & (idx >= max(1, idx_start - n_after_peak))

    return fit_mask


def fit_power_law_tail(k, y, n_after_peak=2, min_points=4):
    """
    Fit a power law y ~ A * k^exp to the high-k tail of y.

    The fit starts 2 points after the maximum of y and uses the rest
    of the available k-range. The fit is done in log-log space:
        log(y) = log(A) + exp * log(k)

    Returns
    -------
    exp_fit : float
        Fitted exponent.
    A_fit : float
        Fitted prefactor.
    tail_mask : boolean array
        Mask of the points used in the fit, in the coordinates of k and y.
    """
    valid = np.isfinite(k) & np.isfinite(y) & (k > 0) & (y > 0)

    k_valid = k[valid]
    y_valid = y[valid]

    if len(k_valid) < min_points:
        return np.nan, np.nan, np.zeros_like(k, dtype=bool)

    peak_idx = np.argmax(y_valid)
    start_idx = peak_idx + n_after_peak

    if start_idx >= len(k_valid) - min_points:
        start_idx = max(0, len(k_valid) - min_points)

    k_tail = k_valid[start_idx:]
    y_tail = y_valid[start_idx:]

    if len(k_tail) < 2:
        return np.nan, np.nan, np.zeros_like(k, dtype=bool)

    coeffs = np.polyfit(np.log(k_tail), np.log(y_tail), 1)
    exp_fit = coeffs[0]
    logA_fit = coeffs[1]
    A_fit = np.exp(logA_fit)

    tail_mask = np.zeros_like(k, dtype=bool)
    valid_indices = np.where(valid)[0]
    tail_indices = valid_indices[start_idx:]
    tail_mask[tail_indices] = True

    return exp_fit, A_fit, tail_mask


def add_fitted_powerlaw_reference(ax, k, y, exp_fit, tail_mask, 
                                  vertical_shift=2.5, label=None):
    """
    Add a reference line y_ref ~ k^exp_fit over the tail region.
    The line is shifted upward by 'vertical_shift' to avoid overlap.
    """
    if np.isnan(exp_fit):
        return

    if np.sum(tail_mask) < 2:
        return

    k_tail = k[tail_mask]
    y_tail = y[tail_mask]

    valid = np.isfinite(k_tail) & np.isfinite(y_tail) & (k_tail > 0) & (y_tail > 0)
    k_tail = k_tail[valid]
    y_tail = y_tail[valid]

    if len(k_tail) < 2:
        return

    # Anchor the guide line near the first point of the fitted tail,
    # but place it somewhat above the data/theory.
    k0 = k_tail[0]
    y0 = vertical_shift * y_tail[0]

    y_ref = y0 * (k_tail / k0) ** exp_fit

    ax.plot(k_tail, y_ref, "k:", lw=lw_ref, label=label)


# ---------- figure ----------
fig, axes = plt.subplots(
    nrows=2,
    ncols=4,
    figsize=(16, 8),
    sharex=True,
    sharey=True
)

for row_idx, Dv_idx in enumerate(Dv_indices):

    Dv = Dv_space[Dv_idx]

    for regime_idx, regime in enumerate(regimes):

        ax = axes[row_idx, regime_idx]

        param_panel = param.copy()
        param_panel["kappa"] = kappas[regime_idx]
        param_panel["D_v"] = Dv

        power_spectra = PS_matrix[Dv_idx, regime_idx]
        G_AB = GAB_matrix[Dv_idx, regime_idx]

        # compute theory
        S_AA_theory = np.zeros(len(k_bins), dtype=complex)
        S_AB_theory = np.zeros(len(k_bins), dtype=complex)
        S_BB_theory = np.zeros(len(k_bins), dtype=complex)

        phi_A_mean = phi_run[0, averaging_bp:, :].mean()
        phi_B_mean = phi_run[1, averaging_bp:, :].mean()

        for i, k in enumerate(k_bins):
            structure_factors = structure_factor(
                k,
                phi_A_mean,
                phi_B_mean,
                param_panel
            )
            S_AA_theory[i] = structure_factors[0, 0]
            S_AB_theory[i] = structure_factors[0, 1]
            S_BB_theory[i] = structure_factors[1, 1]

        # masks
        k_mask = get_k_mask(k_bins, k_max_cutoff=k_max_cutoff)
        fit_mask = get_dynamic_fit_mask(
            k_bins, k_mask, power_spectra, G_AB, n_after_peak=2
        )

        # optionally also remove the first displayed theory points from the fit
        theory_indices_all = np.where(k_mask)[0]
        if len(theory_indices_all) > theory_skip:
            fit_mask[theory_indices_all[:theory_skip]] = False

        # log-space amplitude fit
        theory_fit = np.vstack([
            S_AA_theory[fit_mask].real,
            S_BB_theory[fit_mask].real,
            np.abs(S_AB_theory[fit_mask])
        ])

        data_fit = np.vstack([
            power_spectra[0, fit_mask],
            power_spectra[1, fit_mask],
            np.abs(G_AB[fit_mask])
        ])

        amp = fit_amplitude_logspace(theory_fit, data_fit)

        # plotting mask
        plot_mask = k_mask
        k_plot = k_bins[plot_mask]

        # theory display mask: skip first few displayed theory points
        theory_indices = np.where(plot_mask)[0]
        theory_plot_indices = theory_indices[theory_skip:]

        # ---------- DATA: markers only ----------
        ax.plot(
            k_plot, power_spectra[0, plot_mask],
            linestyle="None", marker="o", color="tab:blue",
            ms=ms, label=r"$S_{AA}$"
        )
        ax.plot(
            k_plot, power_spectra[1, plot_mask],
            linestyle="None", marker="s", color="tab:orange",
            ms=ms, label=r"$S_{BB}$"
        )
        ax.plot(
            k_plot, power_spectra[2, plot_mask],
            linestyle="None", marker="^", color="tab:green",
            ms=ms, label=r"$S_{00}$"
        )
        ax.plot(
            k_plot, np.abs(G_AB[plot_mask]),
            linestyle="None", marker="D", color="tab:red",
            ms=ms, label=r"$|S_{AB}|$"
        )

        # ---------- THEORY ----------
        ax.plot(
            k_bins[theory_plot_indices],
            amp * S_AA_theory[theory_plot_indices].real,
            "-", color="tab:blue", lw=lw_th, label=r"$S_{AA}$ theory"
        )
        ax.plot(
            k_bins[theory_plot_indices],
            amp * S_BB_theory[theory_plot_indices].real,
            "-", color="tab:orange", lw=lw_th, label=r"$S_{BB}$ theory"
        )
        ax.plot(
            k_bins[theory_plot_indices],
            np.abs(amp * S_AB_theory[theory_plot_indices]),
            "-", color="tab:red", lw=lw_th, label=r"$|S_{AB}|$ theory"
        )

        # ---------- fit S_AA tail exponent and add reference ----------
        SAA_plot = power_spectra[0, plot_mask]

        exp_fit, A_fit, tail_mask_local = fit_power_law_tail(
            k_plot, SAA_plot, n_after_peak=5, min_points=4
        )

        add_fitted_powerlaw_reference(
            ax,
            k_plot,
            SAA_plot,
            exp_fit,
            tail_mask_local,
            vertical_shift=2.5,
            label=r"$\sim k^{\mathrm{exp}}$" if (row_idx == 0 and regime_idx == 0) else None
        )

        # add text with fitted exponent
        if np.isfinite(exp_fit):
            exp_text = fr"$\sim k^{{{exp_fit:.2f}}}$"
        else:
            exp_text = r"$\sim k^{\mathrm{nan}}$"

        ax.text(
            0.75, 0.65,
            exp_text,
            transform=ax.transAxes,
            fontsize=text_fs,
            va="top",
            ha="left"
        )

        # axes formatting
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_box_aspect(1)

        if row_idx == 0:
            ax.set_title(regime)

        if regime_idx == 0:
            ax.set_ylabel(fr"$D_v = {param_panel['D_v']}$" + "\nPower spectrum")

        if row_idx == 1:
            ax.set_xlabel(r"$k$")

        ax.tick_params(axis='both', which='both', labelsize=tick_fs)

# global legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="center right",
    bbox_to_anchor=(1.05, 0.5),
    frameon=False,
    fontsize=legend_fs
)

# fig.suptitle("Power spectra across regimes and $D_v$", fontsize=title_fs + 2) 

# plt.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.savefig("images/Schelling_Voter_PS_FV.pdf")
plt.close()
