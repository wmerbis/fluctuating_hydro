import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import gaussian_filter1d
from tqdm.auto import trange
from numba import njit
from dct_spectrum import power_spectrum_dct_2d
import time

# DCT analysis version of ABM_original.py. Microscopic dynamics are unchanged.
# Speed changes: reuse event arrays, update neighbor counts locally, and
# precompute logistic move rates. Event rules and ordering are preserved.
# The simulation uses periodic neighbours; its fields are analysed in a DCT-II
# basis with FV lattice wavenumbers. This does not impose Neumann dynamics.
# Keep dct_spectrum.py alongside this file. Legacy periodic theory is not plotted.

FS_TITLE = 22
FS_LABEL = 20
FS_TICK  = 20
FS_CBAR  = 20
# ---------------- Parameters ----------------------
where_to_start =5
repetition = 1   #  use 1 repetition for animation
M = 9000   # number of snapshots to average over
Nx = 128
Ny = 128
N = Nx * Ny
max_step=10000*N
n_discard = 5   # discard first 5 saved snapshots
snapshot_interval = N # snapshot_interval = max(1, max_step // M)# max_step/M
# GIF playback only; all snapshots remain available to DCT analysis and fitting.
GIF_MAX_FRAMES = 300  # at most 30 seconds at 10 fps
GIF_FPS = 10
fit_bounds = (1e5, np.inf) 
Da = 1
DA = Da   # diffusion of species A
DB = Da   # diffusion of species B
beta = 3
betaA = beta
betaB = beta
vacant = 0.3
kappa_aa = 0.6
kappa_bb = 0.6
kappa_ab = -0.4
kappa_ba = -0.4
Dv = 0
lambda_a = Dv
lambda_b = Dv

# ------------------------- DCT analysis settings --------------------------------
Lx=12.8
Ly=12.8
h = Lx/Nx
dx = Lx/Nx
dy = Ly/Ny
DCT_NUM_BINS = 50
DCT_CHANNELS = ("AA", "BB", "00", "AB", "sym", "pol", "occ")
# Match power_spectrum_dct_2d defaults exactly: centred, orthonormal DCT-II,
# FV lattice k, and radial bins that include the centred DC cell in bin zero.
# Empty bins remain NaN. No additional factor of N is applied to DCT powers.

#----------------------------------------helper functions--------------------------

def fit_power_law_time(t, y):
    """
    Fit y ~ C * t^slope on log-log axes.
    Returns:
        alpha  such that y ~ C * t^(-alpha)
        C
        slope  in log(y) = slope*log(t) + intercept
    """
    mask = np.isfinite(t) & np.isfinite(y) & (t > 0) & (y > 0)
    t_fit = t[mask]
    y_fit = y[mask]

    if len(t_fit) < 2:
        return np.nan, np.nan, np.nan

    logt = np.log(t_fit)
    logy = np.log(y_fit)

    slope, intercept = np.polyfit(logt, logy, 1)
    C = np.exp(intercept)
    alpha = -slope
    return alpha, C, slope

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

#generate initially a pattern to look for the travelling phase

def make_wave_pattern(Nx, Ny, M,freq_x, freq_y, smoothness):
    # if freq_x and freq_y large enough, we get patches 
    X, Y = np.meshgrid(np.linspace(0, 2*np.pi*freq_x, Nx), np.linspace(0, 2*np.pi*freq_y, Ny))
    wave = np.sin(X + 0.5*Y) + 0.5*np.sin(1.5*X - 0.3*Y) + 0.2*np.random.randn(Ny, Nx)
    smooth = gaussian_filter1d(wave, sigma=smoothness, axis=1)
    flat =smooth.flatten()
    y0 = np.zeros_like(flat)
    abs_sorted_indices = np.argsort(np.abs(flat))
    # Indices of elements that will be nonzero
    nonzero_indices = abs_sorted_indices[M:]  
    # Assign ±1 based on sign of original values
    y0[nonzero_indices] = np.sign(flat[nonzero_indices])
    return y0.flatten()

#----------------------------------------Agent-based simulation--------------------------
# --- Graph / lattice generation ---
def generate_graph(Nx, Ny):
    N = Nx*Ny
    neighbors = np.zeros((N, 4), dtype=int)

    def node(x, y): return x + Nx * y

    for x in range(Nx):
        for y in range(Ny):
            i = node(x, y)
            right = node((x+1)%Nx, y)
            left  = node((x-1)%Nx, y)
            up    = node(x, (y+1)%Ny)
            down  = node(x, (y-1)%Ny)
            neighbors[i] = [right, left, up, down]
    return neighbors

@njit
def _initial_neighbor_counts(x, neighbors):
    n_plus = np.zeros(x.size, dtype=np.int64)
    n_minus = np.zeros(x.size, dtype=np.int64)
    for i in range(x.size):
        for k in range(neighbors.shape[1]):
            v = x[neighbors[i, k]]
            if v == 1:
                n_plus[i] += 1
            elif v == -1:
                n_minus[i] += 1
    return n_plus, n_minus


@njit
def _incoming_neighbors(neighbors):
    """Reverse adjacency, including multiplicities, for exact local updates."""
    N, z = neighbors.shape
    starts = np.zeros(N + 1, dtype=np.int64)
    for i in range(N):
        for k in range(z):
            starts[neighbors[i, k] + 1] += 1
    for i in range(N):
        starts[i + 1] += starts[i]
    incoming = np.empty(N * z, dtype=np.int64)
    cursor = starts[:-1].copy()
    for i in range(N):
        for k in range(z):
            j = neighbors[i, k]
            incoming[cursor[j]] = i
            cursor[j] += 1
    return starts, incoming


@njit
def _set_state_and_update_counts(x, site, value, starts, incoming, n_plus, n_minus):
    """Only sites that list this site as a neighbor need new counts."""
    old = x[site]
    plus_change = int(value == 1) - int(old == 1)
    minus_change = int(value == -1) - int(old == -1)
    x[site] = value
    for index in range(starts[site], starts[site + 1]):
        i = incoming[index]
        n_plus[i] += plus_change
        n_minus[i] += minus_change


@njit
def _movement_rate_table(z, beta, Da, kappa_aa, kappa_bb, kappa_ab, kappa_ba):
    """Tabulate exactly the same logistic rates for every integer count change."""
    offset = z + 1
    width = 2 * z + 2
    table = np.empty((2, width, width), dtype=np.float64)
    for species in range(2):
        ka = kappa_aa if species == 0 else kappa_ba
        kb = kappa_ab if species == 0 else kappa_bb
        for a in range(width):
            for b in range(width):
                delta = ka * (a - offset) + kb * (b - offset)
                table[species, a, b] = 2.0 * Da / (1.0 + np.exp(-beta * delta))
    return table


@njit
def _fill_event_buffers(x, neighbors, n_plus, n_minus, table,
                        lambda_a, lambda_b, Schelling,
                        rates, owners, destinations, voter_reaction):
    """Fill reusable arrays in the same event order as the reference code."""
    z = neighbors.shape[1]
    offset = z + 1
    n_moves = 0
    n_voter_entries = 0
    for i in range(x.size):
        xi = x[i]
        if xi == 0:
            continue
        for k in range(z):
            j = neighbors[i, k]
            xj = x[j]
            if xi == 1 and xj == -1:
                voter_reaction[n_voter_entries] = i
                voter_reaction[n_voter_entries + 1] = j
                n_voter_entries += 2
            elif Schelling and xj == 0:
                owners[n_moves] = i
                destinations[n_moves] = j
                if xi == 1:
                    a = n_plus[j] - 1 - n_plus[i] + offset
                    b = n_minus[j] - n_minus[i] + offset
                    rates[n_moves] = table[0, a, b]
                else:
                    a = n_plus[j] - n_plus[i] + offset
                    b = n_minus[j] - 1 - n_minus[i] + offset
                    rates[n_moves] = table[1, a, b]
                n_moves += 1
    for p in range(0, n_voter_entries, 2):
        rates[n_moves + p] = lambda_b
        rates[n_moves + p + 1] = lambda_a
    return n_moves, n_voter_entries


@njit
def simulation_with_snapshots(time, y0, neighbors, beta, Da,
                              kappa_aa, kappa_bb, kappa_ab, kappa_ba,
                              lambda_a, lambda_b,
                              snapshot_interval=snapshot_interval, Schelling=True):
    """Exact Gillespie event rules with reusable buffers and local count updates.

    Candidate events retain the reference ordering: vacancy moves first, then
    paired voter flips. Snapshot spacing remains a count of accepted events.
    The Gamma parameters remain unused, as in the supplied utility calculation.
    """
    t = time[0]
    x = y0.copy()
    snapshots = [x.copy().reshape(Ny, Nx)]
    snap_times = [t]
    N, z = neighbors.shape
    n_plus, n_minus = _initial_neighbor_counts(x, neighbors)
    starts, incoming = _incoming_neighbors(neighbors)
    table = _movement_rate_table(z, beta, Da, kappa_aa, kappa_bb, kappa_ab, kappa_ba)

    # Worst-case capacities; only filled prefixes are read each event.
    owners = np.empty(N * z, dtype=np.int64)
    destinations = np.empty(N * z, dtype=np.int64)
    voter_reaction = np.empty(2 * N * z, dtype=np.int64)
    rates = np.empty(3 * N * z, dtype=np.float64)

    step = 0
    while step < max_step:
        n_moves, n_voter_entries = _fill_event_buffers(
            x, neighbors, n_plus, n_minus, table, lambda_a, lambda_b,
            Schelling, rates, owners, destinations, voter_reaction)
        total_len = n_moves + n_voter_entries
        total_rates = np.sum(rates[:total_len])
        if total_rates == 0:
            break
        r_thresh = np.random.rand() * total_rates
        cumulative = 0.0
        selected_move = 0
        for i in range(total_len):
            cumulative += rates[i]
            if cumulative >= r_thresh:
                selected_move = i
                break

        if selected_move < n_moves:
            origin = owners[selected_move]
            destination = destinations[selected_move]
            spin = x[origin]
            _set_state_and_update_counts(x, destination, spin, starts, incoming, n_plus, n_minus)
            _set_state_and_update_counts(x, origin, 0, starts, incoming, n_plus, n_minus)
        else:
            v_index = selected_move - n_moves
            site = voter_reaction[v_index]
            value = -1 if v_index % 2 == 0 else 1
            _set_state_and_update_counts(x, site, value, starts, incoming, n_plus, n_minus)

        t += -np.log(np.random.rand()) / total_rates
        step += 1
        if step % snapshot_interval == 0:
            snapshots.append(x.copy().reshape(Ny, Nx))
            snap_times.append(t)
    return snapshots, np.array(snap_times), t

# ------------------------------------ Run a single simulation  --------------------

neighbors = generate_graph(Nx, Ny)
#random initial condition
y0 = np.zeros(N)
nonzero_indices = np.random.choice(N, N-int(vacant*N), replace=False)
y0[nonzero_indices] = np.random.choice([-1,1], size=N-int(vacant*N))

# if (kappa_ab*kappa_ba) < 0:
#     #prepatterned condition
#     y0 = make_wave_pattern(Nx, Ny,int(vacant*N), freq_x=4, freq_y=4, smoothness=3)

t0 = time.time()
snapshots, snap_times, trun = simulation_with_snapshots(np.linspace(0,  N/10, N), y0, neighbors,  beta, Da, kappa_aa, kappa_bb, kappa_ab, kappa_ba, lambda_a, lambda_b)
t1 = time.time()
print(trun)
print(f"Getting patterns in {t1-t0:.2f}s")

#--------------------------------------------------- Animation -------------------------------

def select_animation_frames(snap_times, max_frames):
    """Select nearest saved states on a uniform physical-time target grid.

    Saved states generally do not occur exactly at the target times. Keep
    repeated indices (if any) to preserve playback timing; do not interpolate
    the categorical states or modify the full history used for analysis.
    """
    times = np.asarray(snap_times, dtype=float)
    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("Animation times must be a nonempty, finite 1D array.")
    if np.any(np.diff(times) < 0):
        raise ValueError("Animation times must be nondecreasing.")
    if int(max_frames) != max_frames or max_frames < 2:
        raise ValueError("GIF_MAX_FRAMES must be an integer of at least 2.")
    if times.size == 1 or times[-1] == times[0]:
        return times[-1:].copy(), np.array([times.size - 1], dtype=int)

    frame_times = np.linspace(times[0], times[-1], min(int(max_frames), times.size))
    right = np.clip(np.searchsorted(times, frame_times, side="left"), 1, times.size - 1)
    left = right - 1
    # For an exact tie, use the earlier saved state.
    frame_indices = np.where(
        frame_times - times[left] <= times[right] - frame_times, left, right)
    frame_indices[0] = 0
    frame_indices[-1] = times.size - 1
    return frame_times, frame_indices


gif_times, gif_indices = select_animation_frames(snap_times, GIF_MAX_FRAMES)
print(f"GIF: {len(gif_indices)} frames at {GIF_FPS} fps "
      f"({len(gif_indices) / GIF_FPS:.1f} s), "
      f"{np.unique(gif_indices).size} distinct snapshots selected from {len(snapshots)} "
      "on a uniform physical-time target grid.")

fig, ax = plt.subplots()
cmap = plt.get_cmap('bwr', 3)  # blue-white-red
im = ax.imshow(snapshots[gif_indices[0]], cmap=cmap, vmin=-1, vmax=1)
ax.set_title(fr"$\beta = {beta},\,  \kappa aa = {kappa_aa},\, \kappa ab = {kappa_ab},\, \kappa ba = {kappa_ba}, \, Da = {Da},\, Dv={Dv}$")
plt.colorbar(im, ax=ax, ticks=[-1,0,1], label='State')
time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

def update(frame):
    snapshot_index = gif_indices[frame]
    im.set_array(snapshots[snapshot_index])
    time_text.set_text(f"t = {snap_times[snapshot_index]:.6g}")
    return [im, time_text]

update(0)
anim = FuncAnimation(fig, update, frames=len(gif_indices), interval=1000 / GIF_FPS, blit=True)
# --- Export as GIF ---
gif_writer = PillowWriter(fps=GIF_FPS)
anim.save(f"DCT_Beta{beta}_kappa_aa{kappa_aa}_kappa_ab{kappa_ab}_kappa_ba{kappa_ba}_Dv{Dv}_DA{Da}_movie.gif", writer=gif_writer)
plt.show()


# ----------------------- DCT spectra and characteristic k ----------------------

def dct_characteristic_wavenumbers(k_centers, radial_power):
    """Return the radial-bin-weighted mean and peak k for a nonnegative spectrum.

    This follows the original characteristic-k convention: each equally wide
    radial bin contributes its average power, without weighting by its number
    of modes. It is not the mean over every individual 2D mode. Zero-power or
    empty spectra have undefined characteristic k and return NaN.
    """
    k = np.asarray(k_centers, dtype=float)
    power = np.asarray(radial_power, dtype=float)
    valid = np.isfinite(k) & np.isfinite(power) & (k > 0)
    if not np.any(valid):
        return np.nan, np.nan
    k = k[valid]
    power = np.maximum(power[valid], 0.0)
    total = np.sum(power)
    if not np.isfinite(total) or total <= 0:
        return np.nan, np.nan
    # np.argmax gives the smallest-k bin for an exactly tied peak.
    return np.sum(k * power) / total, k[np.argmax(power)]


def analyze_dct_snapshots(snapshots, snap_times, *, domain_lengths,
                          n_discard=5, M=3600, num_bins=50):
    """Analyse one saved frame at a time, keeping only small radial histories.

    All frames contribute to the characteristic-k history. The time-averaged
    spectra retain the original choice: the last M frames remaining after
    n_discard frames are discarded. Saved frames receive equal weight; this is
    an average in accepted-event sampling, not a physical-time-weighted average.

    The DCT helper expects spatial axes (x,y), whereas simulator snapshots use
    (y,x), so each indicator field is transposed before calling the helper.
    """
    nframes = len(snapshots)
    times = np.asarray(snap_times, dtype=float)
    if nframes == 0 or times.shape != (nframes,):
        raise ValueError("Snapshots and snap_times must have the same nonzero length.")
    if not np.all(np.isfinite(times)):
        raise ValueError("snap_times must be finite.")
    if int(n_discard) != n_discard or not 0 <= n_discard < nframes:
        raise ValueError("n_discard must leave at least one snapshot.")
    if int(M) != M or M < 1:
        raise ValueError("M must be a positive integer.")
    if int(num_bins) != num_bins or num_bins < 1:
        raise ValueError("num_bins must be a positive integer.")
    n_discard, M, num_bins = int(n_discard), int(M), int(num_bins)
    take = min(M, nframes - n_discard)
    average_start = nframes - take
    shape_yx = np.asarray(snapshots[0]).shape
    if len(shape_yx) != 2:
        raise ValueError("Each snapshot must be a two-dimensional state array.")
    radial_history = np.full((nframes, len(DCT_CHANNELS), num_bins), np.nan)
    kmean_history = np.full((nframes, len(DCT_CHANNELS)), np.nan)
    kpeak_history = np.full_like(kmean_history, np.nan)
    radial_sum = np.zeros((len(DCT_CHANNELS), num_bins))
    power_sum = np.zeros((len(DCT_CHANNELS), shape_yx[1], shape_yx[0]))

    for frame_index, snap in enumerate(snapshots):
        snap = np.asarray(snap)
        if snap.shape != shape_yx or not np.all(np.isin(snap, (-1, 0, 1))):
            raise ValueError("Snapshots must have a fixed shape and states -1, 0, +1.")
        fields = np.stack(((snap == 1).T, (snap == -1).T)).astype(float)
        fields = fields[:, np.newaxis, :, :]
        (k_centers, autos, cross, auto_2d, cross_2d,
         k_2d) = power_spectrum_dct_2d(
            fields, domain_lengths, bp=0, centered=True, num_bins=num_bins,
            averaged=True, use_lattice_k=True,
        )
        radial = np.stack((
            autos[0], autos[1], autos[2], cross,
            autos[0] + autos[1],
            np.maximum(autos[0] + autos[1] - 2 * cross, 0.0),
            np.maximum(autos[0] + autos[1] + 2 * cross, 0.0),
        ))
        power = np.stack((
            auto_2d[0], auto_2d[1], auto_2d[2], cross_2d,
            auto_2d[0] + auto_2d[1],
            np.maximum(auto_2d[0] + auto_2d[1] - 2 * cross_2d, 0.0),
            np.maximum(auto_2d[0] + auto_2d[1] + 2 * cross_2d, 0.0),
        ))
        radial_history[frame_index] = radial
        for channel_index, channel in enumerate(DCT_CHANNELS):
            # A signed cross-spectrum is not a nonnegative weight distribution.
            if channel != "AB":
                (kmean_history[frame_index, channel_index],
                 kpeak_history[frame_index, channel_index]) = (
                    dct_characteristic_wavenumbers(k_centers, radial[channel_index])
                )
        if frame_index >= average_start:
            radial_sum += radial
            power_sum += power

    return {
        "dct_channels": np.asarray(DCT_CHANNELS),
        "dct_k_centers": k_centers,
        "dct_k_2d": k_2d,
        "dct_snap_times": times,
        "dct_radial_spectra_tidx": radial_history,
        "dct_kmean_tidx": kmean_history,
        "dct_kpeak_tidx": kpeak_history,
        "dct_mean_radial_spectra": radial_sum / take,
        "dct_mean_power_2d": power_sum / take,
        "dct_average_indices": np.arange(average_start, nframes),
        "dct_n_take": np.asarray(take),
        "dct_n_discard": np.asarray(n_discard),
        "dct_num_bins": np.asarray(num_bins),
        "dct_centered": np.asarray(True),
        "dct_use_lattice_k": np.asarray(True),
        "dct_domain_lengths": np.asarray(domain_lengths, dtype=float),
        "dct_spatial_axis_order": np.asarray("x,y"),
        "dct_transform": np.asarray("DCT-II, norm=ortho"),
        "dct_average_weighting": np.asarray("equal saved-frame weights"),
        "dct_boundary_note": np.asarray(
            "DCT analysis basis applied to unchanged periodic ABM trajectories"
        ),
    }


dct_results = analyze_dct_snapshots(
    snapshots, snap_times, domain_lengths=(Lx, Ly), n_discard=n_discard, M=M,
    num_bins=DCT_NUM_BINS,
)
k_vals = dct_results["dct_k_centers"]
take = int(dct_results["dct_n_take"])
SA, SB, S0, SAB, S_sym, S_pol, S_occ = dct_results["dct_mean_radial_spectra"]
_, k0 = dct_characteristic_wavenumbers(k_vals, S_pol)

# Preserve the original spectral-tail selection rule, now explicitly on DCT k.
finite_pol = np.flatnonzero(np.isfinite(S_pol) & (S_pol > 0))
if finite_pol.size:
    peak_index = finite_pol[np.argmax(S_pol[finite_pol])]
    idx_low = peak_index + where_to_start
else:
    idx_low = len(k_vals)
idx_high = -2
k_fit = k_vals[idx_low:idx_high]
slope_A, _ = loglog_slope(k_fit, SA[idx_low:idx_high])
slope_B, _ = loglog_slope(k_fit, SB[idx_low:idx_high])
slope_0, _ = loglog_slope(k_fit, S0[idx_low:idx_high])
slope_AB, _ = loglog_slope(k_fit, np.abs(SAB[idx_low:idx_high]))
slope_sym, _ = loglog_slope(k_fit, S_sym[idx_low:idx_high])
slope_pol, _ = loglog_slope(k_fit, S_pol[idx_low:idx_high])
slope_occ, _ = loglog_slope(k_fit, S_occ[idx_low:idx_high])

# Characteristic-k plots use the full saved history, independently of M.
# Fit both channels from t=10^5 onward; use the same window in the saved results and plot.

time_fit = np.full((2, 3), np.nan)  # columns: alpha, C, log-log slope
fit_counts = np.zeros(2, dtype=int)
for fit_index, channel in enumerate(("pol", "occ")):
    channel_index = DCT_CHANNELS.index(channel)
    k_history = dct_results["dct_kmean_tidx"][:, channel_index]
    valid = (np.isfinite(snap_times) & np.isfinite(k_history)
             & (snap_times >= fit_bounds[0]) & (snap_times <= fit_bounds[1])
             & (snap_times > 0) & (k_history > 0))
    fit_counts[fit_index] = np.count_nonzero(valid)
    time_fit[fit_index] = fit_power_law_time(snap_times[valid], k_history[valid])
    print(f"DCT {channel}: time-fit points={fit_counts[fit_index]}, "
          f"alpha={time_fit[fit_index, 0]:.6g} in {fit_bounds}")

dct_results.update({
    "dct_spectral_tail_exponents": np.asarray((
        slope_A, slope_B, slope_0, slope_AB, slope_sym, slope_pol, slope_occ)),
    "dct_spectral_fit_indices": np.arange(len(k_vals))[idx_low:idx_high],
    "dct_time_fit_channels": np.asarray(("pol", "occ")),
    "dct_time_fit_columns": np.asarray(("alpha", "C", "loglog_slope")),
    "dct_time_fit_values": time_fit,
    "dct_time_fit_counts": fit_counts,
    "dct_time_fit_bounds": np.asarray(fit_bounds),
})

output_stem = (f"DCT_Beta{beta}_kappa_aa{kappa_aa}_kappa_ab{kappa_ab}"
               f"_kappa_ba{kappa_ba}_Da_{Da:.3f}_Dv_{Dv:.3f}")
# Existing raw fields and scalars are retained. k0 and slopeA/B/AB now describe
# DCT spectra; the DCT-specific filename and explicit metadata distinguish them.
np.savez(
    f"scan_{output_stem}.npz",
    snapshots=snapshots, snap_times=snap_times, n_take=take,
    k0=k0, slopeA=slope_A, slopeB=slope_B, slopeAB=slope_AB,
    M=M, Nx=Nx, Ny=Ny, max_step=max_step, beta=beta, vacant=vacant,
    kappa_ab=kappa_ab, kappa_ba=kappa_ba, kappa_aa=kappa_aa,
    kappa_bb=kappa_bb,  h=h, Lx=Lx, Ly=Ly,
    Da=Da, Dv=Dv, snapshot_interval=snapshot_interval,
    **dct_results,
)

fig_k, ax_k = plt.subplots(figsize=(8, 6))
for channel in ("pol", "occ"):
    values = dct_results["dct_kmean_tidx"][:, DCT_CHANNELS.index(channel)]
    valid = (np.isfinite(snap_times) & np.isfinite(values)
             & (values > 0) & (snap_times > 0))
    data_line, = ax_k.loglog(snap_times[valid], values[valid], label=f"DCT S_{channel}")
    fit_valid = valid & (snap_times >= fit_bounds[0]) & (snap_times <= fit_bounds[1])
    alpha, C, slope = fit_power_law_time(snap_times[fit_valid], values[fit_valid])
    print(f"DCT S_{channel}: fit for t >= {fit_bounds[0]:g}, points={np.count_nonzero(fit_valid)}, "
          f"alpha={alpha:.6g}, C={C:.6g}, slope={slope:.6g}; k_mean = C * t**(-alpha)")
    if np.all(np.isfinite((alpha, C, slope))) and C > 0:
        fit_times = snap_times[fit_valid]
        ax_k.loglog(fit_times, C * fit_times**slope, "--", color=data_line.get_color(),
                    label=fr"S_{channel} fit: $\alpha={alpha:.3f}$")
ax_k.set_xlabel("Physical time", fontsize=FS_LABEL)
ax_k.set_ylabel(r"$\langle\hat{k}\rangle$", fontsize=FS_LABEL)
ax_k.set_title(f"DCT-II / FV lattice k; fits for t >= {fit_bounds[0]:g}")
ax_k.legend()
fig_k.tight_layout()
fig_k.savefig(f"{output_stem}_kmean.png", dpi=300)
fig_k.savefig(f"{output_stem}_kmean.pdf")

fig_s, axes = plt.subplots(2, 2, figsize=(13, 10))
image = axes[0, 0].imshow(snapshots[-1], cmap="bwr", vmin=-1, vmax=1)
axes[0, 0].set_title("Final state (periodic ABM)")
fig_s.colorbar(image, ax=axes[0, 0], ticks=(-1, 0, 1))
for channel in ("AA", "BB", "00"):
    values = dct_results["dct_mean_radial_spectra"][DCT_CHANNELS.index(channel)]
    valid = np.isfinite(values) & (values > 0) & (k_vals > 0)
    valid[0] = False
    axes[0, 1].loglog(k_vals[valid], values[valid], "o-", label=f"S_{channel}")
axes[0, 1].set_title("DCT auto-spectra")
for channel in ("sym", "pol", "occ"):
    values = dct_results["dct_mean_radial_spectra"][DCT_CHANNELS.index(channel)]
    valid = np.isfinite(values) & (values > 0) & (k_vals > 0)
    valid[0] = False
    axes[1, 0].loglog(k_vals[valid], values[valid], "o-", label=f"S_{channel}")
axes[1, 0].set_title("DCT composite spectra")
valid_cross = np.isfinite(SAB) & (k_vals > 0)
valid_cross[0] = False
axes[1, 1].semilogx(k_vals[valid_cross], SAB[valid_cross], "o-", label="S_AB (signed)")
axes[1, 1].axhline(0, color="grey", linewidth=0.8)
axes[1, 1].set_title("DCT cross-spectrum")
for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
    ax.set_xlabel(r"FV lattice wavenumber $\hat{k}$")
    ax.set_ylabel("Mean DCT power per radial bin")
    ax.legend()
avg_indices = dct_results["dct_average_indices"]
fig_s.suptitle(
    f"DCT-II / ortho; {take} equally weighted saved frames\n"
    f"t={snap_times[avg_indices[0]]:.6g} to {snap_times[avg_indices[-1]]:.6g}; "
    "no periodic-theory overlay"
)
fig_s.tight_layout()
fig_s.savefig(f"{output_stem}_spectra.png", dpi=300)
fig_s.savefig(f"{output_stem}_spectra.pdf")
plt.show()
