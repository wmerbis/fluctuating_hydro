import os
import sys
import numpy as np
from numba import njit, prange, typed
from numpy.fft import fft2, ifft2, fftshift

outdir = "results_kappa_scan_h01_Dv01"
os.makedirs(outdir, exist_ok=True)

h  = 0.1
Dv = 0.1
# --- Fixed Parameters ---
where_to_start =5
kappa_vals = np.linspace(-1.0, 1.0, 11)   # 11×11 grid
n_k = len(kappa_vals)
n_rep = 10
M = 50   # number of snapshots to keep for taking average 
Nx = 50
Ny = 50
N = Nx * Ny
max_step=100*N #h smaller one needs to run longer!
snapshot_interval = 5 
gamma = 0.2
DA = gamma/(2*h*h)   # diffusion of species A
DB = gamma/(2*h*h)   # diffusion of species B
betaA = 1/DA
betaB = 1/DB
vacant = 0.3
kappa_aa = 0.6
kappa_bb = 0.6
lambda_a = Dv/(h*h) 
lambda_b = Dv/(h*h)
Gamm=1
Gamma_aa = Gamm/(h*h)
Gamma_bb = Gamm/(h*h)
Gamma_ab = 0 
Gamma_ba = 0

# define physical k axes (if domain lengths Lx,Ly, and spacing dx and dy)
dx = h #Lx/Nx in general, if we know the physical length Lx, and the number of houses along one axis, Nx then dx= Lx/Nx, the same for dy=Ly/Ny 
dy = h #Ly/Ny
kx = 2*np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2*np.pi * np.fft.fftfreq(Ny, d=dy)
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

#----------------------------------------Agent-based simulation--------------------------
# --- Graph / lattice generation ---
def generate_graph(Nx, Ny):
    N = Nx*Ny
    A = np.zeros((N, N), np.uint8)
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
            for j in neighbors[i]:
                A[i,j] = A[j,i] = 1
    return A, neighbors

# # --- Utility difference ---
@njit(parallel=True)
def utility_and_hypothetical_utility_flat(x, neighbors,
                                          kappa_aa, kappa_bb, kappa_ab, kappa_ba, Gamma_aa, Gamma_ab, Gamma_ba, Gamma_bb):
    """
    this function returns flattened move data:
      - delta_flat: float64[:]  (Δπ for each possible move)
      - owner: 1d int array where owner[pos] = origin node that owns flattened move pos
      - vac_indices: int64[:]   (destination node index for each move)
      - vac_starts: int64[:]    (prefix array of length N+1; that marks where each i's moves live in, namely, range(vac_starts[i], vac_starts[i+1]) 
    """
    N = x.shape[0]
    z = neighbors.shape[1]

    # 1) Count neighbor types & empty counts
    n_plus = np.zeros(N, dtype=np.int64)
    n_minus = np.zeros(N, dtype=np.int64)
    empty_counts = np.zeros(N, dtype=np.int64)
    total_pairs = 0
    for i in prange(N):
        plus = 0
        minus = 0
        empty = 0
        for k in range(z):
            j = neighbors[i, k]
            v = x[j]
            if v == 1:
                plus += 1
            elif v == -1:
                minus += 1
                if x[i] == 1:
                    total_pairs += 1
            else:
                # v == 0
                empty += 1
        n_plus[i] = plus
        n_minus[i] = minus
        empty_counts[i] = empty

    # 1.1) make the 1-d array for all ordered pair of x[i] = 1 (origin) and x[j] =-1 (target)
    voter_reaction = np.empty(2 * total_pairs, dtype=np.int64)
    p = 0
    for i in range(N):
        if x[i] == 1:
            for k in range(z):
                j = neighbors[i, k]
                if x[j] == -1:
                    voter_reaction[p] = i
                    voter_reaction[p + 1] = j
                    p += 2

    # 2) Compute Laplacians ---
    laplacian_plus = np.zeros(N, dtype=np.float64)
    laplacian_minus = np.zeros(N, dtype=np.float64)

    for i in prange(N):
        if x[i] ==1:
            laplacian_plus[i] = -4.0 + n_plus[i]
            laplacian_minus[i] = n_minus[i]
        elif x[i] ==-1:
            laplacian_minus[i] = -4.0 + n_minus[i]
            laplacian_plus[i] = n_plus[i]
        else:
            laplacian_minus[i] =-4+ n_minus[i]
            laplacian_plus[i] =-4+ n_plus[i]

    # 3) compute pi_plus, pi_minus, pi
    pi_plus = np.empty(N, dtype=np.float64)
    pi_minus = np.empty(N, dtype=np.float64)
    pi = np.empty(N, dtype=np.float64)
    for i in prange(N):
        pi_plus[i] = kappa_aa * n_plus[i] + kappa_ab * n_minus[i] + Gamma_aa*laplacian_plus[i] +Gamma_ab*laplacian_minus[i]
        pi_minus[i] = kappa_ba * n_plus[i] + kappa_bb * n_minus[i] + Gamma_bb*laplacian_minus[i] +  Gamma_ba*laplacian_plus[i]
    for i in prange(N):
        xi = x[i]
        if xi == 1:
            pi[i] = pi_plus[i]
        elif xi == -1:
            pi[i] = pi_minus[i]
        else:
            pi[i] = 0.0

    # 3) Flatten all vacancies into arrays
    total_vac = 0 #total number of vacants of all the nodes
    for i in range(N):
        total_vac += empty_counts[i]

    vac_indices = np.empty(total_vac, dtype=np.int64)  # flattened index for all possible destination vacants
    delta_flat = np.empty(total_vac, dtype=np.float64) # Δπ per move
    owner      = np.empty(total_vac, dtype=np.int64)   # owner (origin) of the moves
    vac_starts = np.empty(N + 1, dtype=np.int64)

    pos = 0 #total number of vacants of all the "occupied" nodes
    for i in range(N):
        vac_starts[i] = pos
        if x[i] == 0:
            # if origin empty, we will not add moves for it
            continue
        # for each neighbor j that is empty, record a move i -> j
        for k in range(z):
            j = neighbors[i, k]
            if x[j] == 0:
                vac_indices[pos] = j
                owner[pos]    = i
                # contribution if moved to j depends on what spin i has
                if x[i] == 1: #we have not consider the new local configuration as the move happens
                    pi_plus_vacant = pi_plus[j] #- (kappa_aa + Gamma_aa)  # 5 comes from -4 *(n==1) and -1 of reducing 1 plus meighbor
                    delta_flat[pos] = pi_plus_vacant - pi[i]
                else:
                    # x[i] == -1 expected (we skipped x[i]==0 above)
                    pi_minus_vacant = pi_minus[j] # - (kappa_bb + Gamma_bb)
                    delta_flat[pos] = pi_minus_vacant- pi[i]
                pos += 1

    vac_starts[N] = pos
    # NOTE: as some origin nodes x[i] were empty, pos should often be fewer than total_vac
    # If pos < total_vac, we return only the filled prefix
    if pos < total_vac:
        # resize arrays to actual length pos (Numba supports creating new arrays)
        out_vac_indices = np.empty(pos, dtype=np.int64)
        out_delta_flat = np.empty(pos, dtype=np.float64)
        out_owner = np.empty(pos, dtype=np.int64)

        for i in range(pos):
            out_vac_indices[i] = vac_indices[i]
            out_delta_flat[i] = delta_flat[i]
            out_owner[i]      = owner[i]

        owner       =  out_owner
        vac_indices = out_vac_indices
        delta_flat = out_delta_flat
    return delta_flat, vac_indices, vac_starts,  owner, voter_reaction

@njit(parallel=True)
def compute_move_rates_flat(delta_flat, gamma):
    """
      - rates: 1d array of acceptance rates (len == len(delta_flat))
    """
    total_moves = delta_flat.shape[0]
    rates = np.empty(total_moves, dtype=np.float64)
    beta = 2.0 / gamma
    rates= gamma / (1.0 + np.exp(-(beta) * delta_flat))
    return rates

@njit(parallel=True)
def compute_voter_rates(voter_reaction, lambda_a, lambda_b):
    """
    - Compute acceptance rates for all neighbor voter reactions.
    - voter_reaction : int64[:] (flattened [i0, j0, i1, j1, ...])
    - Returns
    -------
    voter_rates : float64[:]
        Flattened array with 2 entries per (i,j) pair:
        [rate_for_i_flip, rate_for_j_flip, ...]
    """
    n_pairs = voter_reaction.shape[0] // 2
    voter_rates = np.empty(2 * n_pairs, dtype=np.float64)

    for p in prange(n_pairs):
        # origin and target indices
        i = voter_reaction[2 * p]      # so voter_reaction[0::2] → all origin nodes (x=1)
        j = voter_reaction[2 * p + 1]  # so voter_reaction[1::2] → all corresponding targets (x=-1)

        # assign rates (for every ordered pair (i,j), we store 1 entries in voter_rates)
        # p is the index of the pair in voter_reaction
        voter_rates[2 * p]     = lambda_b  # i: x[i] = 1 → -1 so origin node flips with probability λ_b
        voter_rates[2 * p + 1] = lambda_a  # j: x[j] = -1 → 1 so target node flips with probability λ_a

    return voter_rates

# --- Main Simulation ---
@njit
def simulation_with_snapshots(time, y0, neighbors, gamma, kappa_aa, kappa_bb, kappa_ab, kappa_ba, lambda_a, lambda_b , snapshot_interval=snapshot_interval):
    t = time[0]
    tmax = time[-1]
    record_start = max_step - M * snapshot_interval
    x = y0.copy()
    snapshots = np.zeros((M, Ny, Nx), dtype=np.int8)
    snap_count = 0
    step = 0
    while step < max_step:
        delta_flat, vac_indices, vac_starts, owner, voter_reaction = utility_and_hypothetical_utility_flat(x.astype(np.int64),
                                                                                    neighbors.astype(np.int64),
                                                                                    kappa_aa, kappa_bb, kappa_ab, kappa_ba,  Gamma_aa, Gamma_ab, Gamma_ba, Gamma_bb)

        rates = compute_move_rates_flat(delta_flat,  gamma)
        voter_rates = compute_voter_rates(voter_reaction, lambda_a, lambda_b)
        # Combine into one array
        total_len = rates.shape[0] + voter_rates.shape[0]
        rates_new = np.empty(total_len, dtype=np.float64)
        for i in range(rates.shape[0]):
            rates_new[i] = rates[i]
        for i in range(voter_rates.shape[0]):
            rates_new[rates.shape[0] + i] = voter_rates[i]

        total_rates = np.sum(rates_new)
        if total_rates == 0:
            break
        #selected_move = np.random.choice(len(rates), p=rates/total_rates) #O(N)
        #method with #O(log(N))
        # r = np.random.rand()             
        # threshold = r * total_rates
        # selected_move = np.searchsorted(np.cumsum(rates), threshold)
        r_thresh = np.random.rand() * total_rates
        cumulative = 0.0
        selected_move = 0
        for i in range(rates_new.shape[0]):
            cumulative += rates_new[i]
            if cumulative >= r_thresh:
                selected_move = i
                break
        #Update
        if selected_move < rates.shape[0]:
            node_i = owner[selected_move]
            move_to = vac_indices[selected_move]   # destination node directly from flattened array
            x[move_to] = x[node_i]
            x[node_i] = 0
        else:
            v_index = selected_move - len(rates) # v_index is the index within the voter_rates array
            pair_index = v_index // 2
            i = voter_reaction[2 * pair_index]
            j = voter_reaction[2 * pair_index + 1]

            if v_index % 2 == 0:
                # Origin-side flip: +1 → -1
                x[i] = -1
            else:
                # Target-side flip: -1 → +1
                x[j] = 1

        t += -np.log(np.random.rand()) / total_rates
        step += 1
        # option A: every step is recorded once step >= max_step-M
        if step >= max_step - M:
            snapshots[snap_count % M] = x.reshape(Ny, Nx)
            snap_count += 1
        # option B: if you want “last M snapshots spaced every Δ=snapshot_interval steps”
        # if step >= record_start and (step - record_start) % snapshot_interval == 0:
        #     snapshots[snap_count] = x.reshape(Ny, Nx)
        #     snap_count += 1

    # Return only the snapshots actually filled (last M)
    n_filled = min(snap_count, M)

    return snapshots[:n_filled], t

#------------index computation-----------
def mean_relative_entropy(phi):
    phi0 = 1 - np.sum(phi, axis=0)
    global_dist = np.array([phi[0].mean(), phi[1].mean(), phi0.mean()])
    
    # Ensure no division by zero or log of zero
    phi_combined = np.vstack([phi[0], phi[1], phi0]).reshape((3,)+ phi0.shape)
    global_dist = np.clip(global_dist, 1e-10, None)
    phi_combined = np.clip(phi_combined, 1e-10, None)
    Sglobal = - np.sum(global_dist * np.log(global_dist))
    
    kl_divergence = np.sum(phi_combined * np.log(phi_combined / global_dist.reshape((3,)+ len(phi0.shape)*(1,))), axis=0)
    mean_kl_divergence = np.mean(kl_divergence)/Sglobal
    
    return mean_kl_divergence

def dissimilarity(phi):#block_spin
    phi0 = 1 - np.sum(phi,axis=0)
    global_dist = np.array([phi[0].mean(), phi[1].mean(), phi0.mean()])
    D = np.mean(np.abs(phi[0] - global_dist[0]))/global_dist[0]
    D += np.mean(np.abs(phi[1] - global_dist[1]))/global_dist[1]
    D += np.mean(np.abs(phi0 - global_dist[2]))/global_dist[2]
    return D/2

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
from scipy.ndimage import gaussian_filter1d

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

# --- Parameter scan setup ---
A, neighbors = generate_graph(Nx, Ny)

# ================= SLURM JOB MAPPING =================
job_id = 1# int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

#if job_id < 0 or job_id >= n_k * n_k:
#    raise RuntimeError("Invalid SLURM_ARRAY_TASK_ID")

i = job_id // n_k
j = job_id % n_k

kappa_ab = kappa_vals[i]
kappa_ba = kappa_vals[j]

print(f"[JOB {job_id}] κ_ab={kappa_ab:.3f}, κ_ba={kappa_ba:.3f}")

# ensure to have an array of independent jobs
np.random.seed(12345 + job_id)

# =====================================================
snapshots_all = np.zeros((n_rep, M, Ny, Nx), dtype=np.int8)
n_take_list = np.zeros(n_rep, dtype=np.int32)

k0_list = []
slopeA_list = []
slopeB_list = []
slope0_list = []
slopeAB_list = []
H_list = []
D_list = []
#S_AA_k_list = []
#S_BB_k_list = []
#S_AB_k_list = []

for rep in range(n_rep):

    # random initial condition
    y0 = np.zeros(N)
    nonzero_indices = np.random.choice(N, N-int(vacant*N), replace=False)
    y0[nonzero_indices] = np.random.choice([-1,1], size=N-int(vacant*N))

    if (kappa_ab * kappa_ba) < 0:
        y0 = make_wave_pattern(
            Nx, Ny, int(vacant*N),
            freq_x=4, freq_y=4, smoothness=3
        )

    snapshots, _ = simulation_with_snapshots(
        np.linspace(0, N/10, N),
        y0, neighbors, gamma,
        kappa_aa, kappa_bb, kappa_ab, kappa_ba,
        lambda_a, lambda_b
    )

    #### Power spectrum computed over M snapshots
    take = min(M, len(snapshots))  # take the mininum between the required last M snapshots and the real length of snapshot
    snapshots_all[rep, :take] = snapshots[-take:]
    n_take_list[rep] = take

    S_AA_acc = np.zeros((Ny, Nx), dtype=np.float64)
    S_BB_acc = np.zeros((Ny, Nx), dtype=np.float64)
    S_AB_acc = np.zeros((Ny, Nx), dtype=np.float64)
    S_00_acc = np.zeros((Ny, Nx), dtype=np.float64)
    H_field  = 0
    D_field  = 0

    for snap in snapshots[-take:]: 

        #indicator fields
        rhoA = (snap == 1).astype(float)
        rhoB = (snap == -1).astype(float)
        rho0 = (snap == 0).astype(float)

        stack_field = np.stack([rhoA, rhoB])
        H_field += mean_relative_entropy(stack_field)
        D_field += dissimilarity(stack_field)

        #mean (global) densities
        a0 = rhoA.mean()
        b0 = rhoB.mean()
        #print("Mean densities: a0=", a0, " b0=", b0)

        #fluctuations
        dA = rhoA - a0
        dB = rhoB - b0
        d0 = rho0 - rho0.mean()

        # FFTs
        dA_k = fft2(dA)
        dB_k = fft2(dB)
        d0_k = fft2(d0)
        # accumulate structure factors over the last M snapshots
        S_AA_acc += (dA_k * np.conj(dA_k)).real / N
        S_BB_acc += (dB_k * np.conj(dB_k)).real / N
        S_00_acc += (d0_k * np.conj(d0_k)).real / N
        S_AB_acc += np.real(dA_k * np.conj(dB_k)) / N  # complex in general; take real part for radial averaging

    # average over snapshots
    H_field = H_field/take
    D_field = D_field/take
    S_AA_k = S_AA_acc / take
    S_BB_k = S_BB_acc / take
    S_AB_k = S_AB_acc / take
    S_00_k = S_00_acc / take

    # flatten for radial averaging
    S_AA_flat = S_AA_k.ravel().real
    S_BB_flat = S_BB_k.ravel().real
    S_00_flat = S_00_k.ravel().real
    S_AB_flat = np.real(S_AB_k.ravel())

    Srad_A = np.bincount(bin_index[valid_bin],weights=S_AA_flat[valid_bin],minlength=nbins)
    Srad_B = np.bincount(bin_index[valid_bin],weights=S_BB_flat[valid_bin],minlength=nbins)
    Srad_0 = np.bincount(bin_index[valid_bin],weights=S_00_flat[valid_bin],minlength=nbins)
    Srad_AB = np.bincount(bin_index[valid_bin],weights=S_AB_flat[valid_bin],minlength=nbins)

    mask = counts > 0
    Srad_A[mask] /= counts[mask]
    Srad_B[mask] /= counts[mask]
    Srad_0[mask] /= counts[mask]
    Srad_AB[mask] /= counts[mask]

    # choose data points to fit: exclude k=0 and any bins with zero count
    k_vals = kcent[mask]
    # exclude k=0 bin, so we pick indices where k>0 (should be all valid except possibly first)
    pos      = k_vals > 0
    if len(pos) == 0:
        raise RuntimeError("No nonzero k bins.")

    k_vals   = k_vals[pos]
    SA       = Srad_A[mask][pos]
    SB       = Srad_B[mask][pos]
    S0       = Srad_0[mask][pos]
    SAB      =  Srad_AB[mask][pos]
    S_sym    = SA + SB

    # peak bin index
    k0_list.append(k_vals[np.argmax(S_sym)])
    H_list.append(H_field) 
    D_list.append(D_field)
    #S_AA_k_list.append(S_AA_k)
    #S_BB_k_list.append(S_BB_k)
    #S_AB_k_list.append(S_AB_k) 

    #computing slope and fit------
    # Use ONLY u-th smallest and v-th largest nonzero k bins:
    # k[ pos[u] : pos[-v] ]  → includes everything in between but excludes the extreme ends
    idx_low = np.argmax(S_sym)+where_to_start      # fifth smallest non-zero
    idx_high = -2    # second largest non-zero
    k_fit    = k_vals[idx_low:idx_high]
    SA_fit   = SA[idx_low:idx_high]
    SB_fit   = SB[idx_low:idx_high]
    SAB_fit  = SAB[idx_low:idx_high]
    S0_fit  = S0[idx_low:idx_high]

    slope_A, _ = loglog_slope(k_fit, SA_fit)
    slope_B, _ = loglog_slope(k_fit, SB_fit)
    slope_0, _ = loglog_slope(k_fit, S0_fit)
    slope_AB, _ = loglog_slope(k_fit, np.abs(SAB_fit))

    slopeA_list.append(slope_A)
    slopeB_list.append(slope_B)
    slope0_list.append(slope_0)
    slopeAB_list.append(slope_AB)

outfile = os.path.join(
    outdir,
    f"scan_kab_{i:02d}_kba_{j:02d}.npz"
)

np.savez(
    outfile,
    snapshots=snapshots_all,
    n_take=n_take_list,
    kappa_ab=kappa_ab,
    kappa_ba=kappa_ba,
    k0=np.nanmean(k0_list),
    slopeA=np.nanmean(slopeA_list),
    slopeB=np.nanmean(slopeB_list),
    slope0=np.nanmean(slope0_list),
    slopeAB=np.nanmean(slopeAB_list),
    H=np.nanmean(H_list),
    D=np.nanmean(D_list),
 
)

print(f"[JOB {job_id}] Saved → {outfile}")
